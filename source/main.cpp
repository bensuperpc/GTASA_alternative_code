#if __has_include(<QString>)
#include <QApplication>
#include <QDebug>
#include <QGuiApplication>
#include <QObject>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>
#include <QQuickWindow>
#include <QtGlobal>

#include "application.h"
#include "asyncimageprovider.h"
#include "customlabel.h"
#include "gta_sa_ui.hpp"
#include "imageprovider.h"
#include "lineseries.h"
#include "tablemodel.h"
#include "utils/utils.h"
#else
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gta_cheat_finder/state/GTA_SA_cheat_finder_virtual.hpp"

#include "gta_cheat_finder/state/GTA_SA_cheat_finder_openmp.hpp"
#include "gta_cheat_finder/state/GTA_SA_cheat_finder_stdthread.hpp"

#if defined(BUILD_WITH_CUDA)
#include "gta_cheat_finder/state/GTA_SA_cheat_finder_cuda.hpp"
#endif

#endif

int main(int argc, char* argv[]) {
    bool cli_only = false;

    std::ios_base::sync_with_stdio(false);  // Improve std::cout speed

    std::vector<std::string> args(argv + 1, argv + argc);

    uint64_t calc_mode = 0;
    uint64_t min_range = 0;
    uint64_t max_range = 0;

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
            if (!(iss >> min_range)) {
                std::cout << "Error, non-numeric character !" << std::endl;
                return EXIT_FAILURE;
            }
        } else if (*i == "--max") {
            std::istringstream iss(*++i);
            if (!(iss >> max_range)) {
                std::cout << "Error, non-numeric character !" << std::endl;
                return EXIT_FAILURE;
            }
        } else if (*i == "--calc-mode") {
            std::istringstream iss(*++i);
            if (!(iss >> calc_mode)) {
                std::cout << "Error, non-numeric character !" << std::endl;
                return EXIT_FAILURE;
            }
        } else if (*i == "--cli") {
            cli_only = true;
        } else {
            std::cout << "Unknown argument: " << *i << std::endl;
        }
    }

    std::unique_ptr<GTA_SA_Virtual> gtaSA;

    switch (calc_mode) {
        case 0: {
            gtaSA = std::move(std::make_unique<GTA_SA_STDTHREAD>());
            break;
        }
        case 1: {
            gtaSA = std::move(std::make_unique<GTA_SA_OPENMP>());
            break;
        }
        case 2: {
#ifdef BUILD_WITH_CUDA
            gtaSA = std::move(std::make_unique<GTA_SA_CUDA>());
#else
            std::cout << "CUDA not supported" << std::endl;
#endif
            break;
        }
        default: {
            gtaSA = std::move(std::make_unique<GTA_SA_STDTHREAD>());
            break;
        }
    }

    if (gtaSA == nullptr) {
        std::cout << "Error, gtaSA == nullptr" << std::endl;
        return EXIT_FAILURE;
    }

    // gtaSA->num_thread
    // gtaSA->cuda_block_size

#if __has_include(<QString>)
    if (cli_only == false) {
        /*
        QString home = qEnvironmentVariable("HOME");

        if (!home.isNull()) {
            qDebug() << "HOME: " << home;
        }
        */
        // QQuickWindow::setGraphicsApi(QSGRendererInterface::OpenGL);

        QApplication app(argc, argv);
        app.setOrganizationName("GTA_SA_Cheat_Finder");
        app.setOrganizationDomain("GTA_SA_Cheat_Finder");
        app.setApplicationName("GTA_SA_Cheat_Finder");
        app.setApplicationVersion("0.1.0");

        QQuickStyle::setStyle("Material");
        QQuickStyle::setFallbackStyle("Material");
        QQmlApplicationEngine engine;

        TableModel tablemodel;

        GTA_SA_UI gta_sa_ui(std::move(gtaSA), tablemodel);

        qmlRegisterSingletonType<GTA_SA_UI>("org.bensuperpc.GTA_SAObjects", 1, 0, "GTA_SASingleton",
                                            [&](QQmlEngine*, QJSEngine*) -> QObject* { return &gta_sa_ui; });
        // import org.bensuperpc.GTA_SAObjects 1.0

        qmlRegisterType<GTA_SA_UI>("org.bensuperpc.GTA_SAObjectType", 1, 0, "GTA_SAType");
        // import org.bensuperpc.GTA_SAObjectType 1.0

        application app_ui;
        QMLREGISTERSINGLETONTYPE(app_ui, "org.bensuperpc.application", 1, 0, "AppSingleton")

        
        QMLREGISTERSINGLETONTYPE(tablemodel, "org.bensuperpc.TableData", 1, 0, "TableDataModel")

        engine.addImageProvider(QLatin1String("sync"), new ImageProvider);
        engine.addImageProvider("async", new AsyncImageProvider);

        LineSeries uiData;
        QMLREGISTERSINGLETONTYPE(uiData, "org.bensuperpc.lineseries", 1, 0, "UIData")

        QStringList monModele;
        for (int i = 0; i < 10; i++) {
            monModele.append("Data " + QString::number(i));
        }
        engine.rootContext()->setContextProperty("monModele", QVariant::fromValue(monModele));

        qmlRegisterType<MyLabel>("org.bensuperpc.MyLabelLib", 1, 0, "MyLabel");

        const QUrl url(u"qrc:/bensuperpc.org/bensuperpc/qml/main.qml"_qs);

        QObject::connect(
            &engine, &QQmlApplicationEngine::objectCreated, &app,
            [url](QObject* obj, const QUrl& objUrl) {
                if (!obj && url == objUrl)
                    QCoreApplication::exit(-1);
            },
            Qt::QueuedConnection);
        engine.load(url);

        return app.exec();
    }
#endif

    // Launch operation
    gtaSA->run();

    // Clear old data
    // gtaSA->clear();
    return 0;
}
