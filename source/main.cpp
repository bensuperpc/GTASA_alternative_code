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
#include "gta_sa_ui.hpp"
#include "imageprovider.h"
#include "utils/utils.h"
#else
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#endif

#include "GTASA_alternative_code/GTA_SA_engine.hpp"

int main(int argc, char* argv[]) {
    bool cli_only = false;

    std::ios_base::sync_with_stdio(false);  // Improve std::cout speed

    std::vector<std::string> args(argv + 1, argv + argc);

    uint64_t calc_mode = 0;
    uint64_t minRange = 0;
    uint64_t maxRange = 0;

    for (auto i = args.begin(); i != args.end(); ++i) {
        if (*i == "-h" || *i == "--help") {
            std::cout << "Syntax: GTASA_alternative_code --min <from (uint64_t)> --max "
                         "<to (uint64_t)>"
                         "--calc-mode <0-2> 0: std::thread, 1: OpenMP, 2: CUDA>"
                      << std::endl;
            return EXIT_SUCCESS;
        }
        if (*i == "--min") {
            std::istringstream iss(*++i);
            if (!(iss >> minRange)) {
                std::cout << "Error, non-numeric character !" << std::endl;
                return EXIT_FAILURE;
            }
        } else if (*i == "--max") {
            std::istringstream iss(*++i);
            if (!(iss >> maxRange)) {
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

    std::unique_ptr<GTA_SA_ENGINE> gta_sa_main = std::make_unique<GTA_SA_ENGINE>();

    if (gta_sa_main == nullptr) {
        std::cout << "Error, gtaSA == nullptr" << std::endl;
        return EXIT_FAILURE;
    }

    switch (calc_mode) {
        case 0: {
            gta_sa_main->swichMode(COMPUTE_TYPE::STDTHREAD);
            break;
        }
        case 1: {
            gta_sa_main->swichMode(COMPUTE_TYPE::OPENMP);
            break;
        }
        case 2: {
#ifdef BUILD_WITH_CUDA
            gta_sa_main->swichMode(COMPUTE_TYPE::CUDA);
#else
            std::cout << "CUDA not supported, switching to STDTHREAD" << std::endl;
            gta_sa_main->swichMode(COMPUTE_TYPE::STDTHREAD);
#endif
            break;
        }
        case 3: {
#ifdef BUILD_WITH_OPENCL
            gta_sa_main->swichMode(COMPUTE_TYPE::OPENCL);
#else
            std::cout << "OPENCL not supported, switching to STDTHREAD" << std::endl;
            gta_sa_main->swichMode(COMPUTE_TYPE::STDTHREAD);
#endif
            break;
        }
        default: {
            std::cout << "Unknown calc mode: " << calc_mode << std::endl;
            break;
        }
    }

        // gtaSA->threadCount
        // gtaSA->cudaBlockSize

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
        app.setOrganizationName("GTASA_alternative_code");
        app.setOrganizationDomain("GTASA_alternative_code");
        app.setApplicationName("GTASA_alternative_code");
        app.setApplicationVersion("0.1.0");

        QQuickStyle::setStyle("Material");
        QQuickStyle::setFallbackStyle("Material");
        QQmlApplicationEngine engine;

        GTA_SA_UI gta_sa_ui(std::move(gta_sa_main));

        qmlRegisterSingletonType<GTA_SA_UI>("org.bensuperpc.GTA_SAObjects", 1, 0, "GTA_SASingleton",
                                            [&](QQmlEngine*, QJSEngine*) -> QObject* { return &gta_sa_ui; });
        // import org.bensuperpc.GTA_SAObjects 1.0

        qmlRegisterType<GTA_SA_UI>("org.bensuperpc.GTA_SAObjectType", 1, 0, "GTA_SAType");
        // import org.bensuperpc.GTA_SAObjectType 1.0

        application app_ui;
        QMLREGISTERSINGLETONTYPE(app_ui, "org.bensuperpc.application", 1, 0, "AppSingleton")

        engine.addImageProvider(QLatin1String("sync"), new ImageProvider);
        engine.addImageProvider("async", new AsyncImageProvider);

        QStringList monModele;
        for (int i = 0; i < 10; i++) {
            monModele.append("Data " + QString::number(i));
        }
        engine.rootContext()->setContextProperty("monModele", QVariant::fromValue(monModele));
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
    gta_sa_main->run();

    // Clear old data
    // gtaSA->clear();
    return 0;
}
