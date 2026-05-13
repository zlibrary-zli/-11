#include <QtCore/QDir>
#include <QtCore/QFile>
#include <QtCore/QFileInfo>
#include <QtCore/QJsonDocument>
#include <QtCore/QJsonObject>
#include <QtCore/QProcess>
#include <QtCore/QUrl>
#include <QtGui/QClipboard>
#include <QtGui/QDesktopServices>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
namespace {
QString findImgdiffExecutable() {
  const auto dir = QCoreApplication::applicationDirPath();
  const QStringList candidates = {
    QDir(dir).filePath("imgdiff"),
    QDir(dir).filePath("imgdiff.exe"),
    "imgdiff"
  };
  for (const auto& c : candidates) {
    if (QFileInfo::exists(c)) return c;
  }
  return {};
}

QString shellQuote(const QString& s) {
  if (s.isEmpty()) return "''";
  if (!s.contains(' ') && !s.contains('"') && !s.contains('\'') && !s.contains('\\')) return s;
  QString out = "'";
  for (const auto ch : s) {
    if (ch == '\'') out += "'\\''";
    else out += ch;
  }
  out += "'";
  return out;
}

QWidget* labeledPathRow(QLineEdit* edit, QPushButton* browseBtn) {
  auto* w = new QWidget;
  auto* l = new QHBoxLayout(w);
  l->setContentsMargins(0, 0, 0, 0);
  l->addWidget(edit, 1);
  l->addWidget(browseBtn, 0);
  return w;
}
}

class MainWindow final : public QMainWindow {
  Q_OBJECT

 public:
  MainWindow() {
    setWindowTitle("imgdiff GUI");
    resize(980, 720);

    auto* root = new QWidget;
    auto* rootLayout = new QVBoxLayout(root);

    auto* title = new QLabel("imgdiff");
    title->setStyleSheet("font-size: 22px; font-weight: 700;");
    rootLayout->addWidget(title);

    tabs_ = new QTabWidget;
    tabs_->addTab(buildSingleTab(), "单对图片");
    tabs_->addTab(buildDirTab(), "目录批处理");
    rootLayout->addWidget(tabs_);

    rootLayout->addWidget(buildParamsGroup());

    auto* btnRow = new QHBoxLayout;
    runBtn_ = new QPushButton("运行");
    copyCmdBtn_ = new QPushButton("复制命令行");
    openOutBtn_ = new QPushButton("打开输出目录");
    btnRow->addWidget(runBtn_);
    btnRow->addWidget(copyCmdBtn_);
    btnRow->addStretch(1);
    btnRow->addWidget(openOutBtn_);
    rootLayout->addLayout(btnRow);

    summary_ = new QLabel;
    summary_->setStyleSheet("color: #334155;");
    rootLayout->addWidget(summary_);

    log_ = new QTextEdit;
    log_->setReadOnly(true);
    log_->setMinimumHeight(220);
    rootLayout->addWidget(log_, 1);

    setCentralWidget(root);

    proc_ = new QProcess(this);
    proc_->setProcessChannelMode(QProcess::SeparateChannels);

    connect(runBtn_, &QPushButton::clicked, this, &MainWindow::run);
    connect(copyCmdBtn_, &QPushButton::clicked, this, &MainWindow::copyCmd);
    connect(openOutBtn_, &QPushButton::clicked, this, &MainWindow::openOutDir);
    connect(proc_, &QProcess::readyReadStandardOutput, this, &MainWindow::onStdout);
    connect(proc_, &QProcess::readyReadStandardError, this, &MainWindow::onStderr);
    connect(proc_, &QProcess::finished, this, &MainWindow::onFinished);
  }

 private slots:
  void run() {
    if (proc_->state() != QProcess::NotRunning) return;

    summary_->clear();
    log_->clear();

    const auto program = findImgdiffExecutable();
    if (program.isEmpty()) {
      QMessageBox::critical(this, "imgdiff", "找不到 imgdiff 可执行文件。请确认已构建 imgdiff，并将 imgdiff 放在 imgdiff_gui 同目录，或确保 PATH 可找到 imgdiff。");
      return;
    }

    const bool singleMode = tabs_->currentIndex() == 0;

    const auto outDir = (singleMode ? singleOutDir_ : dirOutDir_)->text().trimmed();
    if (outDir.isEmpty()) {
      QMessageBox::warning(this, "参数缺失", "请先选择输出目录（--out）。");
      return;
    }

    QStringList args;
    if (singleMode) {
      const auto ref = singleRefPath_->text().trimmed();
      const auto tgt = singleTgtPath_->text().trimmed();
      if (ref.isEmpty() || tgt.isEmpty()) {
        QMessageBox::warning(this, "参数缺失", "请先选择 ref/tgt 文件（--ref / --tgt）。");
        return;
      }
      args << "--ref" << ref << "--tgt" << tgt;
      const auto p = singlePrefix_->text().trimmed();
      if (!p.isEmpty()) args << "--prefix" << p;
    } else {
      const auto refDir = dirRefDir_->text().trimmed();
      const auto tgtDir = dirTgtDir_->text().trimmed();
      if (refDir.isEmpty() || tgtDir.isEmpty()) {
        QMessageBox::warning(this, "参数缺失", "请先选择 ref/tgt 目录（--ref_dir / --tgt_dir）。");
        return;
      }
      args << "--ref_dir" << refDir << "--tgt_dir" << tgtDir;
    }

    args << "--out" << outDir;
    args << "--motion" << motion_->currentData().toString();
    args << "--max_dim" << QString::number(maxDim_->value());
    args << "--ecc_iters" << QString::number(eccIters_->value());
    args << "--ecc_eps" << QString::number(eccEps_->value(), 'g', 10);
    args << "--k_mad" << QString::number(kMad_->value(), 'g', 10);
    args << "--min_area" << QString::number(minArea_->value());
    args << "--open_k" << QString::number(openK_->value());
    args << "--close_k" << QString::number(closeK_->value());
    args << "--alpha" << QString::number(alpha_->value(), 'g', 10);
    args << "--save_overlay" << (saveOverlay_->isChecked() ? "1" : "0");
    args << "--save_mask" << (saveMask_->isChecked() ? "1" : "0");
    args << "--save_regions" << (saveRegions_->isChecked() ? "1" : "0");
    args << "--save_report" << (saveReport_->isChecked() ? "1" : "0");
    args << "--save_aligned" << (saveAligned_->isChecked() ? "1" : "0");

    lastProgram_ = program;
    lastArgs_ = args;

    runBtn_->setEnabled(false);
    appendLine(QString("运行：%1 %2").arg(program, argsToCommand(args)));
    proc_->start(program, args);
    if (!proc_->waitForStarted(3000)) {
      runBtn_->setEnabled(true);
      QMessageBox::critical(this, "启动失败", "无法启动 imgdiff。");
      return;
    }
  }

  void copyCmd() {
    if (lastProgram_.isEmpty()) return;
    const auto cmd = shellQuote(lastProgram_) + " " + argsToCommand(lastArgs_);
    QGuiApplication::clipboard()->setText(cmd);
    appendLine("已复制命令行到剪贴板。");
  }

  void openOutDir() {
    const bool singleMode = tabs_->currentIndex() == 0;
    const auto outDir = (singleMode ? singleOutDir_ : dirOutDir_)->text().trimmed();
    if (outDir.isEmpty()) return;
    QDesktopServices::openUrl(QUrl::fromLocalFile(outDir));
  }

  void onStdout() { appendText(QString::fromLocal8Bit(proc_->readAllStandardOutput())); }
  void onStderr() { appendText(QString::fromLocal8Bit(proc_->readAllStandardError())); }

  void onFinished(int exitCode, QProcess::ExitStatus st) {
    runBtn_->setEnabled(true);
    appendLine(QString("结束：exitCode=%1 status=%2").arg(exitCode).arg(st == QProcess::NormalExit ? "NormalExit" : "CrashExit"));

    if (st != QProcess::NormalExit || exitCode != 0) {
      QMessageBox::warning(this, "运行失败", "imgdiff 运行失败，详情见日志。");
      return;
    }

    if (!saveReport_->isChecked()) return;
    if (tabs_->currentIndex() != 0) return;

    const auto outDir = singleOutDir_->text().trimmed();
    const auto refPath = singleRefPath_->text().trimmed();
    const auto prefix = singlePrefix_->text().trimmed();
    QString name = prefix.isEmpty() ? QFileInfo(refPath).completeBaseName() : prefix;
    const auto reportPath = QDir(outDir).filePath(name + "_report.json");
    if (!QFileInfo::exists(reportPath)) return;

    QFile f(reportPath);
    if (!f.open(QIODevice::ReadOnly)) return;
    const auto doc = QJsonDocument::fromJson(f.readAll());
    if (!doc.isObject()) return;
    const auto o = doc.object();
    const auto thr = o.value("diff_threshold").toDouble();
    const auto pix = o.value("diff_pixels").toInt();
    const auto comps = o.value("diff_components").toInt();
    const auto method = o.value("align_method").toString();
    summary_->setText(QString("对齐=%1  阈值=%2  diff_pixels=%3  diff_components=%4  报告=%5").arg(method).arg(thr).arg(pix).arg(comps).arg(reportPath));
  }

 private:
  QWidget* buildSingleTab() {
    auto* w = new QWidget;
    auto* form = new QFormLayout(w);

    singleRefPath_ = new QLineEdit;
    auto* refBtn = new QPushButton("选择…");
    connect(refBtn, &QPushButton::clicked, this, [this] {
      const auto p = QFileDialog::getOpenFileName(this, "选择 ref 图片", {}, "Images (*.bmp *.png *.jpg *.jpeg *.tif *.tiff);;All Files (*.*)");
      if (!p.isEmpty()) singleRefPath_->setText(p);
    });
    form->addRow("ref 文件", labeledPathRow(singleRefPath_, refBtn));

    singleTgtPath_ = new QLineEdit;
    auto* tgtBtn = new QPushButton("选择…");
    connect(tgtBtn, &QPushButton::clicked, this, [this] {
      const auto p = QFileDialog::getOpenFileName(this, "选择 tgt 图片", {}, "Images (*.bmp *.png *.jpg *.jpeg *.tif *.tiff);;All Files (*.*)");
      if (!p.isEmpty()) singleTgtPath_->setText(p);
    });
    form->addRow("tgt 文件", labeledPathRow(singleTgtPath_, tgtBtn));

    singleOutDir_ = new QLineEdit;
    auto* outBtn = new QPushButton("选择…");
    connect(outBtn, &QPushButton::clicked, this, [this] {
      const auto p = QFileDialog::getExistingDirectory(this, "选择输出目录");
      if (!p.isEmpty()) singleOutDir_->setText(p);
    });
    form->addRow("输出目录", labeledPathRow(singleOutDir_, outBtn));

    singlePrefix_ = new QLineEdit;
    form->addRow("前缀（可选）", singlePrefix_);

    return w;
  }

  QWidget* buildDirTab() {
    auto* w = new QWidget;
    auto* form = new QFormLayout(w);

    dirRefDir_ = new QLineEdit;
    auto* refBtn = new QPushButton("选择…");
    connect(refBtn, &QPushButton::clicked, this, [this] {
      const auto p = QFileDialog::getExistingDirectory(this, "选择 ref 目录");
      if (!p.isEmpty()) dirRefDir_->setText(p);
    });
    form->addRow("ref 目录", labeledPathRow(dirRefDir_, refBtn));

    dirTgtDir_ = new QLineEdit;
    auto* tgtBtn = new QPushButton("选择…");
    connect(tgtBtn, &QPushButton::clicked, this, [this] {
      const auto p = QFileDialog::getExistingDirectory(this, "选择 tgt 目录");
      if (!p.isEmpty()) dirTgtDir_->setText(p);
    });
    form->addRow("tgt 目录", labeledPathRow(dirTgtDir_, tgtBtn));

    dirOutDir_ = new QLineEdit;
    auto* outBtn = new QPushButton("选择…");
    connect(outBtn, &QPushButton::clicked, this, [this] {
      const auto p = QFileDialog::getExistingDirectory(this, "选择输出目录");
      if (!p.isEmpty()) dirOutDir_->setText(p);
    });
    form->addRow("输出目录", labeledPathRow(dirOutDir_, outBtn));

    auto* tip = new QLabel("目录批处理仅匹配同名 .bmp 文件。");
    tip->setStyleSheet("color: #64748b;");
    form->addRow("", tip);

    return w;
  }

  QWidget* buildParamsGroup() {
    auto* g = new QGroupBox("参数");
    auto* form = new QFormLayout(g);

    motion_ = new QComboBox;
    motion_->addItem("euclidean（旋转+平移）", "euclidean");
    motion_->addItem("affine（仿射）", "affine");
    motion_->addItem("translation（仅平移）", "translation");
    form->addRow("motion", motion_);

    maxDim_ = new QSpinBox;
    maxDim_->setRange(64, 20000);
    maxDim_->setValue(2000);
    form->addRow("max_dim", maxDim_);

    eccIters_ = new QSpinBox;
    eccIters_->setRange(1, 20000);
    eccIters_->setValue(200);
    form->addRow("ecc_iters", eccIters_);

    eccEps_ = new QDoubleSpinBox;
    eccEps_->setDecimals(10);
    eccEps_->setRange(1e-12, 1.0);
    eccEps_->setValue(1e-6);
    form->addRow("ecc_eps", eccEps_);

    kMad_ = new QDoubleSpinBox;
    kMad_->setDecimals(4);
    kMad_->setRange(0.0, 100.0);
    kMad_->setValue(6.0);
    form->addRow("k_mad", kMad_);

    minArea_ = new QSpinBox;
    minArea_->setRange(0, 1000000);
    minArea_->setValue(30);
    form->addRow("min_area", minArea_);

    openK_ = new QSpinBox;
    openK_->setRange(0, 255);
    openK_->setValue(3);
    form->addRow("open_k", openK_);

    closeK_ = new QSpinBox;
    closeK_->setRange(0, 255);
    closeK_->setValue(5);
    form->addRow("close_k", closeK_);

    alpha_ = new QDoubleSpinBox;
    alpha_->setDecimals(3);
    alpha_->setRange(0.0, 1.0);
    alpha_->setSingleStep(0.05);
    alpha_->setValue(0.6);
    form->addRow("alpha", alpha_);

    auto* outFlags = new QWidget;
    auto* outLayout = new QHBoxLayout(outFlags);
    outLayout->setContentsMargins(0, 0, 0, 0);
    saveOverlay_ = new QCheckBox("overlay");
    saveMask_ = new QCheckBox("mask");
    saveRegions_ = new QCheckBox("regions.csv");
    saveReport_ = new QCheckBox("report.json");
    saveAligned_ = new QCheckBox("tgt_aligned");
    saveOverlay_->setChecked(true);
    saveMask_->setChecked(true);
    saveRegions_->setChecked(true);
    saveReport_->setChecked(true);
    saveAligned_->setChecked(true);
    outLayout->addWidget(saveOverlay_);
    outLayout->addWidget(saveMask_);
    outLayout->addWidget(saveRegions_);
    outLayout->addWidget(saveReport_);
    outLayout->addWidget(saveAligned_);
    outLayout->addStretch(1);
    form->addRow("输出", outFlags);

    return g;
  }

  void appendText(const QString& s) {
    if (s.isEmpty()) return;
    log_->moveCursor(QTextCursor::End);
    log_->insertPlainText(s);
    log_->moveCursor(QTextCursor::End);
  }

  void appendLine(const QString& s) { appendText(s + "\n"); }

  QString argsToCommand(const QStringList& args) const {
    QStringList parts;
    parts.reserve(args.size());
    for (const auto& a : args) parts << shellQuote(a);
    return parts.join(" ");
  }

  QTabWidget* tabs_{};

  QLineEdit* singleRefPath_{};
  QLineEdit* singleTgtPath_{};
  QLineEdit* singleOutDir_{};
  QLineEdit* singlePrefix_{};

  QLineEdit* dirRefDir_{};
  QLineEdit* dirTgtDir_{};
  QLineEdit* dirOutDir_{};

  QComboBox* motion_{};
  QSpinBox* maxDim_{};
  QSpinBox* eccIters_{};
  QDoubleSpinBox* eccEps_{};
  QDoubleSpinBox* kMad_{};
  QSpinBox* minArea_{};
  QSpinBox* openK_{};
  QSpinBox* closeK_{};
  QDoubleSpinBox* alpha_{};

  QCheckBox* saveOverlay_{};
  QCheckBox* saveMask_{};
  QCheckBox* saveRegions_{};
  QCheckBox* saveReport_{};
  QCheckBox* saveAligned_{};

  QPushButton* runBtn_{};
  QPushButton* copyCmdBtn_{};
  QPushButton* openOutBtn_{};
  QLabel* summary_{};
  QTextEdit* log_{};

  QProcess* proc_{};
  QString lastProgram_;
  QStringList lastArgs_;
};

#include "imgdiff_gui.moc"

int main(int argc, char** argv) {
  QApplication app(argc, argv);
  MainWindow w;
  w.show();
  return app.exec();
}

