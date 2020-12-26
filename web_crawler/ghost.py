import sys
import time
from PySide2.QtWidgets import (
    QApplication,
)
from PySide2.QtWebEngineWidgets import (
    QWebEnginePage,
    QWebEngineSettings,
    QWebEngineView
)
from PySide2.QtCore import (
    QUrl,
    QSize
)

class GhostWebPage(QWebEnginePage):
    """Overrides QtWebKit.QWebPage in order to intercept some graphical
    behaviours like alert(), confirm().
    Also intercepts client side console.log().
    """
    def __init__(self, app, session):
        self.session = session
        super(GhostWebPage, self).__init__()

class Ghost(object):
    """`Ghost` manages a Qt application.
    """
    _app = None
    def __init__(self):
        print("\033[0;36mInitializing QT application\033[0m")
        Ghost._app = QApplication.instance() or QApplication(['ghost'])
    def start(self, **kwargs):
        """Starts a new `Session`.
        """
        return Session(self, **kwargs)
    def exit(self):
        self._app.quit()
        if hasattr(self, 'xvfb'):
            self.xvfb.stop()
    def __del__(self):
        self.exit()

class Session(object):
    """`Session` manages a QWebPage.
    """
    _app = None
    def __init__(self, ghost, viewport_size=(1600, 900), web_page_class=GhostWebPage):
        self.ghost = ghost

        self.html = None
        self.loaded = True
        
        self.page = web_page_class(self.ghost._app, self)
        
        self.page.settings().setAttribute(
            QWebEngineSettings.LocalStorageEnabled, True)
        
        self.page.settings().setAttribute(
            QWebEngineSettings.AutoLoadImages, False)
        self.page.settings().setAttribute(
            QWebEngineSettings.ScrollAnimatorEnabled, True)
        self.page.settings().setAttribute(
            QWebEngineSettings.JavascriptEnabled, True)
        # Page signals
        self.page.loadFinished.connect(self._page_loaded)
        self.page.loadStarted.connect(self._page_load_started)
        self.page.loadProgress.connect(self._page_load_progress)

        class GhostQWebView(QWebEngineView):
            def sizeHint(self):
                return QSize(*viewport_size)

        self.webview = GhostQWebView()
        self.set_viewport_size(*viewport_size)

        self.webview.setPage(self.page)
    def set_viewport_size(self, width, height):
        new_size = QSize(width, height)

        self.webview.resize(new_size)
        self.sleep()
    def open(self, address):
        """Opens a web page.
        """
        self.webview.load(QUrl(address)) 
        self._wait()
    def scroll_to_id(self, id):
        self.page.runJavaScript("""
            (function(){
                window.location.href='#%s'
            })();
        """ % (id,))

    def click(self, selector, btn=0):
        self.page.runJavaScript("""
            (function(){
                var element = document.querySelectorAll(%s);
                var event = new MouseEvent('click', {
                    'view': window,
                    'bubbles': true,
                    'cancelable': true
                });
                return element[%s].dispatchEvent(event);
            })();
        """ % (repr(selector), str(btn)))
    def wait_for_page_loaded(self, selector):
        self.loaded = False
        while not self.loaded:
            if self.exists(selector):
                self.loaded = True
            else:
                self.loaded = False
        return self.html
    def to_html(self):
        self.page.toHtml(self._to_html_callback)
        self._wait()
        return self.html
    def exists(self, selector):
        self.to_html()
        if selector in self.html:
            return True
        else:
            return False
    def show(self):
        self.webview.show()
        self.sleep()
    def sleep(self, value=0.1):
        started_at = time.time()
        while time.time() <= (started_at + value):
            time.sleep(0.01)
            self.ghost._app.processEvents()
    def exit(self):
        """Exits all Qt widgets."""
        self.page.deleteLater()
        self.sleep()
        del self.webview
        del self.page
    def _wait(self):
        self.loaded = False
        while not self.loaded:
            self.sleep()
    def _page_load_started(self):
        """Called back when page load started.
        """
        print("\033[0;32m_page_load_started\033[0m")
        self.loaded = False
    def _page_loaded(self):
        """Called back when page is loaded.
        """
        print("\033[0;32m_page_loaded\033[0m")
        self.loaded = True
    def _page_load_progress(self, progress):
        print("\033[0;32m_page_loaded_progress:\033[0m", progress)
    def _to_html_callback(self, data):
        self.html = data
        self.loaded = True
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()
