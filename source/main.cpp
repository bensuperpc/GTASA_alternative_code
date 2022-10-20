#if __has_include(<QString>)
#  include <QDirIterator>
#  include <QGuiApplication>
#  include <QQmlApplicationEngine>
#  include <QQmlContext>
#  include <QStringList>
#endif

#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "GTA_SA_cheat_finder.hpp"

#if __has_include(<QString>)
#  include "about_compilation.h"
#  include "gta_sa_ui.h"
#  include "qt-utils/TableModel.h"
#endif

auto main(int argc, char* argv[]) -> int
{
  bool cli_only = false;

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
    } else if (*i == "--cli") {
      cli_only = true;
    } else {
      std::cout << "Unknown argument: " << *i << std::endl;
    }
  }

  // gta_sa.num_thread
  // gta_sa.min_range
  // gta_sa.max_range
  // gta_sa.num_thread
  // gta_sa.cuda_block_size
  // gta_sa.calc_mode

#if __has_include(<QString>)
  if (cli_only == false) {
    /*
    QDirIterator it(":", QDirIterator::Subdirectories);
    while (it.hasNext()) {
      qDebug() << it.next();
    }
    */
    QGuiApplication app(argc, argv);

    app.setOrganizationName("GTA_SA_Cheat_Finder");
    app.setOrganizationDomain("GTA_SA_Cheat_Finder");
    app.setApplicationName("GTA_SA_Cheat_Finder");
    app.setApplicationVersion("0.1.0");

    /*
    const QStringList args = QCoreApplication::arguments();

    for (auto i = 0; i < args.size(); ++i) {
      std::cout << args[i].toStdString() << std::endl;
    }
    */

    // Initialize BEFORE class instance
    QQmlApplicationEngine engine;

    about_compilation ab;
    GTA_SA_UI gta_sa_ui;
    gta_sa_ui.set_gta_sa(gta_sa);

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
  }
#endif

  // CLI only or no Qt

  // Launch operation
  gta_sa.run();

  // Clear old data
  // gta_sa.clear();
  return 0;
}
