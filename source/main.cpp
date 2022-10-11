#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#if __has_include(<QApplication>)
// Building with Qt
#  include <QApplication>
#  include <QGuiApplication>
#  include <QQmlApplicationEngine>
#  include <QQmlContext>

#  include "gta_sa_ui.h"
#include "about_compilation.h"
#else
// Not building with Qt
#  include "GTA_SA_cheat_finder.hpp"
#endif

auto main(int argc, char* argv[]) -> int
{
  std::vector<std::string> args(argv + 1, argv + argc);

  // std::ios_base::sync_with_stdio(false);  // Improve std::cout speed

#if __has_include(<QApplication>)

#  if ((QT_VERSION >= QT_VERSION_CHECK(5, 6, 0)) && (QT_VERSION < QT_VERSION_CHECK(6, 0, 0)))
  QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
  QGuiApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#  endif

#  if (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))
  QApplication::setHighDpiScaleFactorRoundingPolicy(Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);
#  endif

  QApplication app(argc, argv);

  about_compilation ac;

  GTA_SA_UI gta_sa_ui;

  QQmlApplicationEngine engine;  // Create Engine AFTER initializing the classes
  QQmlContext* context = engine.rootContext();

  context->setContextProperty("gta_sa", &gta_sa_ui);
  context->setContextProperty("about_compilation", &ac);
  
  context->setContextProperty("qtversion", QString(qVersion()));

  const QUrl url(u"qrc:/bensuperpc.com/qml_files/source/qml/main.qml"_qs);
  QObject::connect(
      &engine,
      &QQmlApplicationEngine::objectCreated,
      &app,
      [url](const QObject* obj, const QUrl& obj_url)
      {
        if (!obj && url == obj_url)
          QCoreApplication::exit(-1);
      },
      Qt::QueuedConnection);
  engine.load(url);

  return app.exec();

#else  // Not building with Qt

  GTA_SA gta_sa;

  for (auto i = args.begin(); i != args.end(); ++i) {
    if (*i == "-h" || *i == "--help") {
      std::cout << "Syntax: GTA_SA_cheat_finder --min <from (uint64_t)> --max "
                   "<to (uint64_t)>"
                   "--calc-mode <0-2> 0: std::thread, 1: OpenMP, 2: CUDA>"
                << std::endl;
      return EXIT_SUCCESS;
    }
    if (*i == "--min") {
      std::istringstream iss(*++i);
      if (!(iss >> gta_sa.min_range)) {
        std::cout << "Error, non-numeric character !" << std::endl;
        return EXIT_FAILURE;
      }
    } else if (*i == "--max") {
      std::istringstream iss(*++i);
      if (!(iss >> gta_sa.max_range)) {
        std::cout << "Error, non-numeric character !" << std::endl;
        return EXIT_FAILURE;
      }
    } else if (*i == "--calc-mode") {
      std::istringstream iss(*++i);
      if (!(iss >> gta_sa.calc_mode)) {
        std::cout << "Error, non-numeric character !" << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  // gta_sa.num_thread
  // gta_sa.min_range
  // gta_sa.max_range
  // gta_sa.num_thread
  // gta_sa.cuda_block_size
  // gta_sa.calc_mode

  // Launch operation
  gta_sa.run();

// Clear old data
// gta_sa.clear();
#endif
  return 0;
}
