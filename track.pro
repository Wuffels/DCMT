#-------------------------------------------------
#
# Project created by QtCreator 2014-07-18T13:21:50
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = track
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += \
    track.cpp \
    in_out.cpp \
    track_clear_function.cpp \
    functions.cpp \
    main0.cpp

HEADERS += \
    trackobject.h \
    in_out.h \
    track_clear_function.h \
    functions.h

OPENCVDIR = "C:/workspace/opencv-2.4.10/opencv/build"

CONFIG(debug, debug|release) {
      OPENCVDIR = $$OPENCVDIR
      CVVER=2410d
      DESTDIR = bin/debug
}
CONFIG(release, debug|release) {
       OPENCVDIR = $$OPENCVDIR
       CVVER=2410
       DESTDIR = bin/release
}

INCLUDEPATH += $${OPENCVDIR}/include

LIBS += -L$${OPENCVDIR}/x86/vc11/lib \
-lopencv_core$${CVVER} \
-lopencv_imgproc$${CVVER} \
-lopencv_highgui$${CVVER} \
-lopencv_calib3d$${CVVER} \
-lopencv_features2d$${CVVER} \
-lopencv_nonfree$${CVVER} \
-lopencv_video$${CVVER} \
-lopencv_flann$${CVVER} \
-lopencv_ml$${CVVER}

DISTFILES += \
    text.txt
