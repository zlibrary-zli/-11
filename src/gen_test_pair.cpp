#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

static void die(const std::string& msg) {
  std::cerr << msg << "\n";
  std::exit(2);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: gen_test_pair <out_dir>\n";
    return 0;
  }

  fs::path outDir(argv[1]);
  std::error_code ec;
  fs::create_directories(outDir, ec);
  if (ec) die("Failed to create output directory");

  int w = 1024;
  int h = 768;

  cv::Mat ref(h, w, CV_16U);
  for (int y = 0; y < h; ++y) {
    uint16_t* row = ref.ptr<uint16_t>(y);
    for (int x = 0; x < w; ++x) {
      double gx = static_cast<double>(x) / static_cast<double>(w - 1);
      double gy = static_cast<double>(y) / static_cast<double>(h - 1);
      double v = 12000.0 + 15000.0 * gx + 8000.0 * gy;
      row[x] = static_cast<uint16_t>(std::clamp(v, 0.0, 65535.0));
    }
  }

  cv::circle(ref, cv::Point(w / 3, h / 2), 120, cv::Scalar(26000), -1, cv::LINE_AA);
  cv::rectangle(ref, cv::Rect(w * 2 / 3 - 90, h / 2 - 60, 180, 120), cv::Scalar(18000), -1, cv::LINE_AA);

  cv::Mat tgt = ref.clone();

  cv::Mat rot = cv::getRotationMatrix2D(cv::Point2f(w * 0.5f, h * 0.5f), 0.6, 1.0);
  rot.at<double>(0, 2) += 2.0;
  rot.at<double>(1, 2) += -3.0;
  cv::warpAffine(tgt, tgt, rot, tgt.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0));

  cv::circle(tgt, cv::Point(w / 2 + 150, h / 2 + 80), 18, cv::Scalar(36000), -1, cv::LINE_AA);

  fs::path refPath = outDir / "ref.png";
  fs::path tgtPath = outDir / "tgt.png";
  if (!cv::imwrite(refPath.string(), ref)) die("Failed to write ref");
  if (!cv::imwrite(tgtPath.string(), tgt)) die("Failed to write tgt");

  std::cout << "Wrote:\n";
  std::cout << "  " << refPath.string() << "\n";
  std::cout << "  " << tgtPath.string() << "\n";
  return 0;
}
