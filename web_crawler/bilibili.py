from ghost import Ghost

ghost = Ghost()
with ghost.start() as session:
    session.open("https://www.bilibili.com/video/BV1zi4y157Jb/#comment") 
    html = session.to_html()
    print("\033[0;35mPage 1\033[0m", '<span class="current">1</span>' in html)
    
    html = session.wait_for_page_loaded('<span class="current">1</span>')
    print("\033[0;35mPage 1\033[0m", '<span class="current">1</span>' in html)
    
    session.click(".paging-box-big .tcd-number")
    html = session.wait_for_page_loaded('<span class="current">2</span>')
    print("\033[0;35mPage 2\033[0m", '<span class="current">2</span>' in html)


