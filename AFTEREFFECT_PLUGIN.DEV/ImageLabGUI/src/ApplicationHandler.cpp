#include "ApplicationHandler.hpp"
#include "ImageLabQTBinding.hpp"

using namespace ImageLabGUI;

// Initialize static members
std::mutex CGUIInterface::s_AppprotectMutex;

// Initialize atomic pointer
std::atomic<CGUIInterface*> CGUIInterface::s_AppInstance{ nullptr };

// Initialize the static QApplication pointer
QApplication* CGUIInterface::s_pluginQtApp = nullptr; 

bool CGUIInterface::s_weCreatedTheQApplication = false;

int CGUIInterface::s_dummyArgc = 0;
char** CGUIInterface::s_dummyArgv = nullptr;


CGUIInterface* CGUIInterface::getInstance()
{
    CGUIInterface* iGuiHandler = s_AppInstance.load(std::memory_order_acquire);
    if (nullptr == iGuiHandler)
    {
        std::lock_guard<std::mutex> myLock(s_AppprotectMutex);
        iGuiHandler = s_AppInstance.load(std::memory_order_relaxed); // Double-check
        if (nullptr == iGuiHandler)
        {
            try
            {
                iGuiHandler = new CGUIInterface(); // Calls private constructor
                s_AppInstance.store(iGuiHandler, std::memory_order_release);
            }
            catch (const std::exception& e)
            {
                qCritical() << "CGUIInterface::getInstance - Failed to create CGUIInterface singleton:" << e.what();
            }
            catch (...)
            {
                qCritical() << "CGUIInterface::getInstance - Failed to create CGUIInterface singleton: Unknown exception.";
            }
        }
    }
    return iGuiHandler;
}

// Private constructor
CGUIInterface::CGUIInterface(void)
{
    if (QCoreApplication::instance() == nullptr)
    {
        try
        {
            // s_pluginQtApp is static, so assign to it directly
            s_pluginQtApp = new QApplication(s_dummyArgc, s_dummyArgv);
            s_weCreatedTheQApplication = true;
        }
        catch (const std::exception& e)
        {
            qCritical() << "CGUIInterface: Failed to create QApplication:" << e.what();
            s_pluginQtApp = nullptr; // Ensure it's null on failure
            throw; // Propagate exception so getInstance() knows creation failed
        }
        catch (...)
        {
            qCritical() << "CGUIInterface: Failed to create QApplication with unknown exception.";
            s_pluginQtApp = nullptr;
            throw;
        }
    }
    else
    {
        qDebug() << "CGUIInterface: Existing QCoreApplication found. Attempting to use it.";
        s_pluginQtApp = qobject_cast<QApplication*>(QCoreApplication::instance());
        s_weCreatedTheQApplication = false;
        if (s_pluginQtApp == nullptr)
        {
            qWarning() << "CGUIInterface: Existing QCoreApplication is not a QApplication. Qt Widgets UI may not function.";
        }
        else
        {
            qDebug() << "CGUIInterface: Successfully using existing QApplication.";
        }
    }

    return;
}

CGUIInterface::~CGUIInterface(void)
{
    // class destructor
    qDebug() << "CGUIInterface destructor called.";
    if (nullptr != s_pluginQtApp && s_weCreatedTheQApplication)
    {
        qDebug() << "CGUIInterface: Deleting the QApplication instance it created.";
        delete s_pluginQtApp;
        s_pluginQtApp = nullptr; // Nullify the static pointer
        s_weCreatedTheQApplication = false;
    }
    else if (s_pluginQtApp != nullptr)
    {
        qDebug() << "CGUIInterface: Not deleting QApplication as this instance did not create it, or it was already cleaned up.";
        // If this class didn't create it, it shouldn't delete it.
        // Also, if another part of the system already deleted s_pluginQtApp, this avoids a double delete.
        // Consider if s_pluginQtApp should be nullified here even if not deleted by this class.
        // s_pluginQtApp = nullptr; // Cautious: might impact other users if they expect it via QCoreApplication::instance()
    }

    return;
}

// Static member function to access the QApplication pointer
QApplication* CGUIInterface::getQtApplication(void)
{
    // We assume getInstance() has been called at least once to initialize s_pluginQtApp.
    if (s_pluginQtApp == nullptr)
    {
        qWarning() << "CGUIInterface::getQtApplication() called but s_pluginQtApp is null. "
            << "Ensure CGUIInterface::getInstance() was called successfully first.";
    }
    return s_pluginQtApp;
}


void CGUIInterface::cleanupInstanceAndQt (void)
{
    // Protect access to s_AppInstance
    std::lock_guard<std::mutex> myLock(s_AppprotectMutex); 
    CGUIInterface* instance = s_AppInstance.load(std::memory_order_relaxed);

    if (instance != nullptr)
    {
        qDebug() << "CGUIInterface::cleanupInstanceAndQt: Deleting CGUIInterface singleton instance.";
        delete instance;
        // Mark singleton as gone
        s_AppInstance.store(nullptr, std::memory_order_release);
    }
    else
    {
        qDebug() << "CGUIInterface::cleanupInstanceAndQt: Singleton instance was already null.";

        if (s_pluginQtApp != nullptr && s_weCreatedTheQApplication)
        {
            qDebug() << "CGUIInterface::cleanupInstanceAndQt: Singleton was null, but found a QApplication this class might have created. Cleaning it up.";
            delete s_pluginQtApp;
            s_pluginQtApp = nullptr;
            s_weCreatedTheQApplication = false;
        }
    }
}

void ImageLabGUI::DeleteQTApplication (void)
{
    return CGUIInterface::cleanupInstanceAndQt();
}


QApplication* AllocQTApplication (void)
{
    return CGUIInterface::getQtApplication();
}



void FreeQTApplication(void)
{
    // nothing;
}