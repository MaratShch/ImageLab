#ifndef __IMAGE_LAB2_APPLICATION_QT_HANDLER_GUI__
#define __IMAGE_LAB2_APPLICATION_QT_HANDLER_GUI__

#include <atomic>
#include <mutex>
#include "ClassRestrictions.hpp"
#include <QApplication>
#include <QDebug> // For logging

namespace ImageLabGUI
{
    class CGUIInterface
    {
        public:

            // Keep your singleton getInstance
            static CGUIInterface* getInstance(); 

            // Static method to get the QApplication pointer
            static QApplication* getQtApplication();

            // Explicitly disallow copying and assignment for a singleton
            CLASS_NON_COPYABLE(CGUIInterface);
            CLASS_NON_MOVABLE(CGUIInterface);

            // New public static method for cleanup
            static void cleanupInstanceAndQt ();

        private:
            CGUIInterface();  // Private constructor
            ~CGUIInterface(); // Private destructor

            static std::atomic<CGUIInterface*> s_AppInstance; // Singleton instance holder
            static std::mutex s_AppprotectMutex;              // Mutex for thread-safe creation

            // This will hold the single QApplication instance for the process
            static QApplication* s_pluginQtApp;
            static bool s_weCreatedTheQApplication; // Flag: did this class create it?

            // Dummy argc, argv if we need to create QApplication
            static int s_dummyArgc;
            static char** s_dummyArgv; // For nullptr style if argc=0
    };

    void DeleteQTApplication (void);

}

#endif // __IMAGE_LAB2_APPLICATION_QT_HANDLER_GUI__