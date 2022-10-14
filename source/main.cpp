#include <QDirIterator>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QStringList>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "GTA_SA_cheat_finder.hpp"
#include "TableModel.h"
#include "gta_sa_ui.h"
#include "about_compilation.h"

auto main(int argc, char* argv[]) -> int
{
  /*
  QDirIterator it(":", QDirIterator::Subdirectories);
  while (it.hasNext()) {
    qDebug() << it.next();
  }
  */

  QGuiApplication app(argc, argv);
  // Initialize BEFORE class instance
  QQmlApplicationEngine engine;

  about_compilation ab;
  GTA_SA_UI gta_sa_ui;
  qmlRegisterSingletonType<GTA_SA_UI>("org.bensuperpc.GTA_SAObjects",
                                      1,
                                      0,
                                      "GTA_SASingleton",
                                      [&](QQmlEngine*, QJSEngine*) -> QObject* { return &gta_sa_ui; });
  // import org.bensuperpc.GTA_SAObjects 1.0

  qmlRegisterType<GTA_SA_UI>("org.bensuperpc.GTA_SAObjectType", 1, 0, "GTA_SAType");
  // import org.bensuperpc.GTA_SAObjectType 1.0

  qmlRegisterSingletonType<TableModel>("org.bensuperpc.TableModelObjects",
                                       1,
                                       0,
                                       "TableModelObjects",
                                       [&](QQmlEngine*, QJSEngine*) -> QObject* { return &gta_sa_ui.tableModel; });
  // import org.bensuperpc.TableModelObjects 1.0

  qmlRegisterSingletonInstance("org.bensuperpc.ABCObjects", 1, 0, "ABCObjects", &ab);

  //engine.rootContext()->setContextProperty("myModel", &gta_sa_ui.tableModel);

  const QUrl url(u"qrc:/bensuperpc.com/qml_files/source/qml/main.qml"_qs);
  QObject::connect(
      &engine,
      &QQmlApplicationEngine::objectCreated,
      &app,
      [url](QObject* obj, const QUrl& objUrl)
      {
        if (!obj && url == objUrl)
          QCoreApplication::exit(-1);
      },
      Qt::QueuedConnection);
  engine.load(url);

  return app.exec();

  GTA_SA gta_sa;

  std::ios_base::sync_with_stdio(false);  // Improve std::cout speed

  std::vector<std::string> args(argv + 1, argv + argc);

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
  return 0;
}
