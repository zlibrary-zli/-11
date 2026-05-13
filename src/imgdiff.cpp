#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

static std::string toLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

static bool isBmpPath(const fs::path& p) {
  auto ext = toLower(p.extension().string());
  return ext == ".bmp";
}

static bool ensureDir(const fs::path& p) {
  std::error_code ec;
  if (fs::exists(p, ec)) return fs::is_directory(p, ec);
  return fs::create_directories(p, ec);
}

static void die(const std::string& msg) {
  std::cerr << msg << "\n";
  std::exit(2);
}

struct PercentileRange {
  double low = 0.0;
  double high = 0.0;
};

static cv::Mat loadImageAnyDepth(const std::string& path) {
  cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
  if (img.empty()) die("Failed to read image: " + path);
  if (img.depth() != CV_8U && img.depth() != CV_16U) die("Unsupported image depth (need 8U or 16U): " + path);
  return img;
}

static cv::Mat toGrayKeepDepth(const cv::Mat& img) {
  if (img.channels() == 1) return img;
  cv::Mat gray;
  if (img.channels() == 3) {
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return gray;
  }
  if (img.channels() == 4) {
    cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    return gray;
  }
  die("Unsupported channel count: " + std::to_string(img.channels()));
  return {};
}

static cv::Mat grayToFloat01(const cv::Mat& gray) {
  double denom = 1.0;
  if (gray.depth() == CV_8U) denom = 255.0;
  if (gray.depth() == CV_16U) denom = 65535.0;
  cv::Mat f;
  gray.convertTo(f, CV_32F, 1.0 / denom);
  return f;
}

static cv::Mat resizeMaxDim(const cv::Mat& img, int maxDim, double* outScale) {
  int w = img.cols;
  int h = img.rows;
  int maxSide = std::max(w, h);
  if (maxSide <= maxDim) {
    if (outScale) *outScale = 1.0;
    return img;
  }
  double scale = static_cast<double>(maxDim) / static_cast<double>(maxSide);
  cv::Mat out;
  cv::resize(img, out, cv::Size(), scale, scale, cv::INTER_AREA);
  if (outScale) *outScale = scale;
  return out;
}

static PercentileRange percentilesSampled(const cv::Mat& gray, double pLow, double pHigh, int step) {
  if (gray.channels() != 1) die("percentilesSampled expects single channel");

  std::vector<uint32_t> values;
  values.reserve(static_cast<size_t>((gray.rows / step + 1) * (gray.cols / step + 1)));

  if (gray.depth() == CV_8U) {
    for (int y = 0; y < gray.rows; y += step) {
      const uint8_t* row = gray.ptr<uint8_t>(y);
      for (int x = 0; x < gray.cols; x += step) values.push_back(row[x]);
    }
  } else if (gray.depth() == CV_16U) {
    for (int y = 0; y < gray.rows; y += step) {
      const uint16_t* row = gray.ptr<uint16_t>(y);
      for (int x = 0; x < gray.cols; x += step) values.push_back(row[x]);
    }
  } else {
    die("Unsupported depth for percentilesSampled");
  }

  if (values.empty()) die("No pixels for percentile calculation");

  std::sort(values.begin(), values.end());
  auto idx = [&](double p) -> size_t {
    double f = p / 100.0;
    double pos = f * static_cast<double>(values.size() - 1);
    return static_cast<size_t>(std::clamp(pos, 0.0, static_cast<double>(values.size() - 1)));
  };

  PercentileRange r;
  r.low = static_cast<double>(values[idx(pLow)]);
  r.high = static_cast<double>(values[idx(pHigh)]);
  if (r.high <= r.low) r.high = r.low + 1.0;
  return r;
}

struct RobustStats {
  double median = 0.0;
  double mad = 0.0;
};

static RobustStats medianAndMadSampled(const cv::Mat& imgFloat, int step) {
  if (imgFloat.channels() != 1 || imgFloat.depth() != CV_32F) die("medianAndMadSampled expects CV_32F single channel");

  std::vector<float> v;
  v.reserve(static_cast<size_t>((imgFloat.rows / step + 1) * (imgFloat.cols / step + 1)));
  for (int y = 0; y < imgFloat.rows; y += step) {
    const float* row = imgFloat.ptr<float>(y);
    for (int x = 0; x < imgFloat.cols; x += step) v.push_back(row[x]);
  }
  if (v.empty()) die("No pixels for robust stats");
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  float med = v[v.size() / 2];
  for (auto& x : v) x = std::abs(x - med);
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  float mad = v[v.size() / 2];
  return RobustStats{static_cast<double>(med), static_cast<double>(mad)};
}

static std::optional<cv::Mat> estimateWarpECC(
  const cv::Mat& refGray,
  const cv::Mat& tgtGray,
  int motionType,
  int maxIters,
  double eps,
  double* outEccScore
) {
  cv::Mat refF = grayToFloat01(refGray);
  cv::Mat tgtF = grayToFloat01(tgtGray);

  cv::GaussianBlur(refF, refF, cv::Size(3, 3), 0);
  cv::GaussianBlur(tgtF, tgtF, cv::Size(3, 3), 0);

  cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
  cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, maxIters, eps);

  try {
    double cc = cv::findTransformECC(refF, tgtF, warp, motionType, criteria);
    if (outEccScore) *outEccScore = cc;
    return warp;
  } catch (const cv::Exception&) {
    return std::nullopt;
  }
}

static cv::Mat estimateWarpFallbackTranslation(const cv::Mat& refGray, const cv::Mat& tgtGray) {
  cv::Mat refF = grayToFloat01(refGray);
  cv::Mat tgtF = grayToFloat01(tgtGray);

  cv::Mat hann;
  cv::createHanningWindow(hann, refF.size(), CV_32F);
  cv::Point2d shift = cv::phaseCorrelate(refF, tgtF, hann);

  cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
  warp.at<float>(0, 2) = static_cast<float>(shift.x);
  warp.at<float>(1, 2) = static_cast<float>(shift.y);
  return warp;
}

static cv::Mat scaleWarpTranslation(const cv::Mat& warpSmall, double invScale) {
  cv::Mat warp = warpSmall.clone();
  warp.at<float>(0, 2) = static_cast<float>(warp.at<float>(0, 2) * invScale);
  warp.at<float>(1, 2) = static_cast<float>(warp.at<float>(1, 2) * invScale);
  return warp;
}

static cv::Mat applyWarpToTarget(const cv::Mat& tgt, const cv::Mat& warp, const cv::Size& outSize) {
  cv::Mat out;
  int flags = cv::INTER_CUBIC + cv::WARP_INVERSE_MAP;
  cv::warpAffine(tgt, out, warp, outSize, flags, cv::BORDER_CONSTANT, cv::Scalar::all(0));
  return out;
}

static cv::Mat matchIntensityLinear16U(const cv::Mat& refGray, const cv::Mat& tgtGrayAligned, int sampleStep) {
  PercentileRange refP = percentilesSampled(refGray, 5.0, 95.0, sampleStep);
  PercentileRange tgtP = percentilesSampled(tgtGrayAligned, 5.0, 95.0, sampleStep);

  double scale = (refP.high - refP.low) / (tgtP.high - tgtP.low);
  double bias = refP.low - scale * tgtP.low;

  cv::Mat tgtF;
  tgtGrayAligned.convertTo(tgtF, CV_32F);
  tgtF = tgtF * static_cast<float>(scale) + static_cast<float>(bias);

  cv::Mat tgtOut;
  if (refGray.depth() == CV_16U) {
    cv::threshold(tgtF, tgtF, 65535.0, 65535.0, cv::THRESH_TRUNC);
    cv::threshold(tgtF, tgtF, 0.0, 0.0, cv::THRESH_TOZERO);
    tgtF.convertTo(tgtOut, CV_16U);
  } else {
    cv::threshold(tgtF, tgtF, 255.0, 255.0, cv::THRESH_TRUNC);
    cv::threshold(tgtF, tgtF, 0.0, 0.0, cv::THRESH_TOZERO);
    tgtF.convertTo(tgtOut, CV_8U);
  }
  return tgtOut;
}

static cv::Mat absDiffFloat(const cv::Mat& aGray, const cv::Mat& bGray) {
  cv::Mat aF, bF;
  aGray.convertTo(aF, CV_32F);
  bGray.convertTo(bF, CV_32F);
  cv::Mat d;
  cv::absdiff(aF, bF, d);
  return d;
}

static double estimateThresholdFromDiff(const cv::Mat& diffF, double kMad, int step) {
  RobustStats rs = medianAndMadSampled(diffF, step);
  double sigma = 1.4826 * rs.mad;
  double thr = rs.median + kMad * sigma;
  if (thr < 1e-6) thr = 1e-6;
  return thr;
}

static cv::Mat buildMask(const cv::Mat& diffF, double thr) {
  cv::Mat mask;
  cv::threshold(diffF, mask, thr, 255.0, cv::THRESH_BINARY);
  mask.convertTo(mask, CV_8U);
  return mask;
}

static cv::Mat morphClean(const cv::Mat& mask, int openK, int closeK) {
  cv::Mat m = mask.clone();
  if (closeK > 0) {
    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closeK, closeK));
    cv::morphologyEx(m, m, cv::MORPH_CLOSE, k);
  }
  if (openK > 0) {
    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(openK, openK));
    cv::morphologyEx(m, m, cv::MORPH_OPEN, k);
  }
  return m;
}

struct MaskStats {
  int64_t pixels = 0;
  int components = 0;
};

struct Region {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int area = 0;
};

static std::tuple<cv::Mat, MaskStats, std::vector<Region>> filterByArea(const cv::Mat& mask, int minArea) {
  cv::Mat labels, stats, centroids;
  int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

  cv::Mat out = cv::Mat::zeros(mask.size(), CV_8U);
  int kept = 0;
  int64_t pix = 0;
  std::vector<Region> regions;
  for (int i = 1; i < n; ++i) {
    int area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (area < minArea) continue;
    regions.push_back(Region{
      stats.at<int>(i, cv::CC_STAT_LEFT),
      stats.at<int>(i, cv::CC_STAT_TOP),
      stats.at<int>(i, cv::CC_STAT_WIDTH),
      stats.at<int>(i, cv::CC_STAT_HEIGHT),
      area,
    });
    cv::Mat sel = labels == i;
    out.setTo(255, sel);
    kept += 1;
    pix += area;
  }
  return {out, MaskStats{pix, kept}, regions};
}

static cv::Mat toneMapTo8U(const cv::Mat& gray, int step) {
  PercentileRange p = percentilesSampled(gray, 2.0, 98.0, step);
  cv::Mat f;
  gray.convertTo(f, CV_32F);
  f = (f - static_cast<float>(p.low)) * (255.0f / static_cast<float>(p.high - p.low));
  cv::threshold(f, f, 255.0, 255.0, cv::THRESH_TRUNC);
  cv::threshold(f, f, 0.0, 0.0, cv::THRESH_TOZERO);
  cv::Mat out;
  f.convertTo(out, CV_8U);
  return out;
}

static double percentileOfFloatSampled(const cv::Mat& imgFloat, double p, int step) {
  if (imgFloat.channels() != 1 || imgFloat.depth() != CV_32F) die("percentileOfFloatSampled expects CV_32F single channel");
  std::vector<float> v;
  v.reserve(static_cast<size_t>((imgFloat.rows / step + 1) * (imgFloat.cols / step + 1)));
  for (int y = 0; y < imgFloat.rows; y += step) {
    const float* row = imgFloat.ptr<float>(y);
    for (int x = 0; x < imgFloat.cols; x += step) v.push_back(row[x]);
  }
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  double f = p / 100.0;
  double pos = f * static_cast<double>(v.size() - 1);
  size_t i = static_cast<size_t>(std::clamp(pos, 0.0, static_cast<double>(v.size() - 1)));
  return static_cast<double>(v[i]);
}

static cv::Mat buildOverlay(const cv::Mat& refGray, const cv::Mat& diffF, const cv::Mat& mask, double alpha) {
  cv::Mat bg8 = toneMapTo8U(refGray, 16);
  cv::Mat bgBgr;
  cv::cvtColor(bg8, bgBgr, cv::COLOR_GRAY2BGR);

  double p99 = percentileOfFloatSampled(diffF, 99.0, 16);
  double denom = std::max(p99, 1e-6);
  cv::Mat diff8F = diffF * static_cast<float>(255.0 / denom);
  cv::threshold(diff8F, diff8F, 255.0, 255.0, cv::THRESH_TRUNC);
  cv::threshold(diff8F, diff8F, 0.0, 0.0, cv::THRESH_TOZERO);
  cv::Mat diff8;
  diff8F.convertTo(diff8, CV_8U);

  cv::Mat heat;
  cv::applyColorMap(diff8, heat, cv::COLORMAP_JET);

  cv::Mat out = bgBgr.clone();
  if (mask.empty()) return out;

  cv::Mat maskF;
  mask.convertTo(maskF, CV_32F, 1.0 / 255.0);
  std::vector<cv::Mat> m3(3, maskF);
  cv::Mat mask3;
  cv::merge(m3, mask3);

  cv::Mat outF, bgF, heatF;
  out.convertTo(outF, CV_32F);
  bgBgr.convertTo(bgF, CV_32F);
  heat.convertTo(heatF, CV_32F);

  cv::Mat blend = bgF.mul(1.0 - alpha * mask3) + heatF.mul(alpha * mask3);
  blend.convertTo(out, CV_8U);
  return out;
}

struct RunConfig {
  std::string refPath;
  std::string tgtPath;
  std::string refDir;
  std::string tgtDir;
  std::string outDir;
  std::string prefix;
  std::string motion = "euclidean";
  int maxDim = 2000;
  int eccIters = 200;
  double eccEps = 1e-6;
  double kMad = 6.0;
  int minArea = 30;
  int openK = 3;
  int closeK = 5;
  double alpha = 0.6;
  bool saveAligned = true;
};

static int motionTypeFromString(const std::string& s) {
  std::string v = toLower(s);
  if (v == "translation") return cv::MOTION_TRANSLATION;
  if (v == "euclidean") return cv::MOTION_EUCLIDEAN;
  if (v == "affine") return cv::MOTION_AFFINE;
  die("Unknown motion type: " + s);
  return cv::MOTION_EUCLIDEAN;
}

static void writeJsonReport(
  const fs::path& path,
  const cv::Mat& warp,
  const std::string& method,
  double eccScore,
  double thr,
  const MaskStats& ms
) {
  std::ofstream f(path);
  if (!f) die("Failed to write: " + path.string());

  f << "{\n";
  f << "  \"align_method\": \"" << method << "\",\n";
  f << "  \"ecc_score\": " << eccScore << ",\n";
  f << "  \"warp\": [\n";
  f << "    [" << warp.at<float>(0, 0) << ", " << warp.at<float>(0, 1) << ", " << warp.at<float>(0, 2) << "],\n";
  f << "    [" << warp.at<float>(1, 0) << ", " << warp.at<float>(1, 1) << ", " << warp.at<float>(1, 2) << "]\n";
  f << "  ],\n";
  f << "  \"diff_threshold\": " << thr << ",\n";
  f << "  \"diff_pixels\": " << ms.pixels << ",\n";
  f << "  \"diff_components\": " << ms.components << "\n";
  f << "}\n";
}

static void writeRegionsCsv(const fs::path& path, const std::vector<Region>& regions) {
  std::ofstream f(path);
  if (!f) die("Failed to write: " + path.string());
  f << "x,y,w,h,area\n";
  for (const auto& r : regions) {
    f << r.x << "," << r.y << "," << r.w << "," << r.h << "," << r.area << "\n";
  }
}

static void processPair(const RunConfig& cfg, const std::string& refPath, const std::string& tgtPath, const std::string& name) {
  cv::Mat refAny = loadImageAnyDepth(refPath);
  cv::Mat tgtAny = loadImageAnyDepth(tgtPath);

  cv::Mat refGray = toGrayKeepDepth(refAny);
  cv::Mat tgtGray = toGrayKeepDepth(tgtAny);
  if (refGray.size() != tgtGray.size()) die("Image size mismatch for pair: " + name);

  cv::Mat refSmallF, tgtSmallF;
  double scale = 1.0;
  {
    cv::Mat refSmall = resizeMaxDim(refGray, cfg.maxDim, &scale);
    cv::Mat tgtSmall = resizeMaxDim(tgtGray, cfg.maxDim, nullptr);
    refSmallF = refSmall;
    tgtSmallF = tgtSmall;
  }

  int motionType = motionTypeFromString(cfg.motion);
  double eccScore = -1.0;
  std::string alignMethod = "ecc";
  cv::Mat warpSmall;

  auto warpOpt = estimateWarpECC(refSmallF, tgtSmallF, motionType, cfg.eccIters, cfg.eccEps, &eccScore);
  if (warpOpt.has_value()) {
    warpSmall = *warpOpt;
  } else {
    alignMethod = "phase_correlation_translation";
    eccScore = -1.0;
    warpSmall = estimateWarpFallbackTranslation(refSmallF, tgtSmallF);
    motionType = cv::MOTION_TRANSLATION;
  }

  double invScale = 1.0 / scale;
  cv::Mat warpFull = scaleWarpTranslation(warpSmall, invScale);

  cv::Mat tgtGrayAligned = applyWarpToTarget(tgtGray, warpFull, refGray.size());

  cv::Mat tgtMatched = matchIntensityLinear16U(refGray, tgtGrayAligned, 16);
  cv::Mat diffF = absDiffFloat(refGray, tgtMatched);

  double thr = estimateThresholdFromDiff(diffF, cfg.kMad, 16);
  cv::Mat mask = buildMask(diffF, thr);
  mask = morphClean(mask, cfg.openK, cfg.closeK);

  auto [mask2, ms, regions] = filterByArea(mask, cfg.minArea);
  cv::Mat overlay = buildOverlay(refGray, diffF, mask2, cfg.alpha);

  fs::path outDir(cfg.outDir);
  ensureDir(outDir);

  fs::path overlayPath = outDir / (name + "_overlay.png");
  fs::path maskPath = outDir / (name + "_mask.png");
  fs::path regionsPath = outDir / (name + "_regions.csv");
  fs::path reportPath = outDir / (name + "_report.json");

  if (!cv::imwrite(overlayPath.string(), overlay)) die("Failed to write: " + overlayPath.string());
  if (!cv::imwrite(maskPath.string(), mask2)) die("Failed to write: " + maskPath.string());
  writeRegionsCsv(regionsPath, regions);

  if (cfg.saveAligned) {
    fs::path alignedPath = outDir / (name + "_tgt_aligned.png");
    cv::Mat aligned8 = toneMapTo8U(tgtGrayAligned, 16);
    if (!cv::imwrite(alignedPath.string(), aligned8)) die("Failed to write: " + alignedPath.string());
  }

  writeJsonReport(reportPath, warpFull, alignMethod, eccScore, thr, ms);
}

static std::unordered_map<std::string, std::string> parseArgs(int argc, char** argv) {
  std::unordered_map<std::string, std::string> m;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0) die("Unknown argument: " + k);
    if (k == "--help" || k == "-h") {
      m["--help"] = "1";
      continue;
    }
    if (i + 1 >= argc) die("Missing value for " + k);
    std::string v = argv[++i];
    m[k] = v;
  }
  return m;
}

static void printUsage() {
  std::cout << "Usage:\n";
  std::cout << "  imgdiff --ref <a.bmp> --tgt <b.bmp> --out <out_dir> [--prefix <name>]\n";
  std::cout << "  imgdiff --ref_dir <dirA> --tgt_dir <dirB> --out <out_dir>\n";
  std::cout << "Options:\n";
  std::cout << "  --motion euclidean|affine|translation\n";
  std::cout << "  --max_dim <int> (alignment downsample max side, default 2000)\n";
  std::cout << "  --ecc_iters <int> (default 200)\n";
  std::cout << "  --ecc_eps <float> (default 1e-6)\n";
  std::cout << "  --k_mad <float> (threshold = median + k*MAD*1.4826, default 6)\n";
  std::cout << "  --min_area <int> (filter small components, default 30)\n";
  std::cout << "  --open_k <int> (morph open kernel size, default 3)\n";
  std::cout << "  --close_k <int> (morph close kernel size, default 5)\n";
  std::cout << "  --alpha <float> (overlay alpha, default 0.6)\n";
  std::cout << "  --save_aligned 0|1 (default 1)\n";
}

static RunConfig buildConfigFromArgs(const std::unordered_map<std::string, std::string>& a) {
  RunConfig cfg;

  auto get = [&](const std::string& k) -> std::optional<std::string> {
    auto it = a.find(k);
    if (it == a.end()) return std::nullopt;
    return it->second;
  };
  auto getInt = [&](const std::string& k, int def) -> int {
    auto v = get(k);
    if (!v) return def;
    return std::stoi(*v);
  };
  auto getDouble = [&](const std::string& k, double def) -> double {
    auto v = get(k);
    if (!v) return def;
    return std::stod(*v);
  };

  cfg.refPath = get("--ref").value_or("");
  cfg.tgtPath = get("--tgt").value_or("");
  cfg.refDir = get("--ref_dir").value_or("");
  cfg.tgtDir = get("--tgt_dir").value_or("");
  cfg.outDir = get("--out").value_or("");
  cfg.prefix = get("--prefix").value_or("");

  cfg.motion = get("--motion").value_or(cfg.motion);
  cfg.maxDim = getInt("--max_dim", cfg.maxDim);
  cfg.eccIters = getInt("--ecc_iters", cfg.eccIters);
  cfg.eccEps = getDouble("--ecc_eps", cfg.eccEps);
  cfg.kMad = getDouble("--k_mad", cfg.kMad);
  cfg.minArea = getInt("--min_area", cfg.minArea);
  cfg.openK = getInt("--open_k", cfg.openK);
  cfg.closeK = getInt("--close_k", cfg.closeK);
  cfg.alpha = getDouble("--alpha", cfg.alpha);
  cfg.saveAligned = getInt("--save_aligned", cfg.saveAligned ? 1 : 0) != 0;

  if (cfg.outDir.empty()) die("Missing required: --out");

  bool fileMode = !cfg.refPath.empty() || !cfg.tgtPath.empty();
  bool dirMode = !cfg.refDir.empty() || !cfg.tgtDir.empty();
  if (fileMode && dirMode) die("Use either --ref/--tgt or --ref_dir/--tgt_dir");

  if (fileMode) {
    if (cfg.refPath.empty() || cfg.tgtPath.empty()) die("Missing required: --ref and --tgt");
  } else if (dirMode) {
    if (cfg.refDir.empty() || cfg.tgtDir.empty()) die("Missing required: --ref_dir and --tgt_dir");
  } else {
    die("Missing required: --ref/--tgt or --ref_dir/--tgt_dir");
  }

  return cfg;
}

int main(int argc, char** argv) {
  if (argc <= 1) {
    printUsage();
    return 0;
  }

  RunConfig cfg;
  try {
    auto args = parseArgs(argc, argv);
    if (args.count("--help")) {
      printUsage();
      return 0;
    }
    cfg = buildConfigFromArgs(args);
  } catch (const std::exception& e) {
    die(std::string("Argument error: ") + e.what());
  }

  if (!cfg.refPath.empty()) {
    std::string name = cfg.prefix;
    if (name.empty()) {
      fs::path p(cfg.refPath);
      name = p.stem().string();
    }
    processPair(cfg, cfg.refPath, cfg.tgtPath, name);
    return 0;
  }

  fs::path refDir(cfg.refDir);
  fs::path tgtDir(cfg.tgtDir);
  if (!fs::exists(refDir) || !fs::is_directory(refDir)) die("Invalid --ref_dir");
  if (!fs::exists(tgtDir) || !fs::is_directory(tgtDir)) die("Invalid --tgt_dir");

  ensureDir(cfg.outDir);

  int processed = 0;
  for (const auto& entry : fs::directory_iterator(refDir)) {
    if (!entry.is_regular_file()) continue;
    fs::path refPath = entry.path();
    if (!isBmpPath(refPath)) continue;

    fs::path tgtPath = tgtDir / refPath.filename();
    if (!fs::exists(tgtPath)) continue;

    std::string name = refPath.stem().string();
    processPair(cfg, refPath.string(), tgtPath.string(), name);
    processed += 1;
  }

  if (processed == 0) die("No matching bmp pairs found");
  return 0;
}
