import time
import re
import sys
from bs4 import BeautifulSoup
import os
import pandas as pd
root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_path) 
from pool import ThreadPool
from ghost import Ghost

class Crawler(object):
    comments = []
    def __init__(self, address, num_pool=2):
        """初始化线层ThreadPool, Ghost抓取数据
        Args:
            address: 网页网址
        """
        self.address = address+"/#comment"
        ghost = Ghost()
        self.session = ghost.start()
        # self.session.show()

        self.session.open(self.address)
        html = self.session.wait_for_page_loaded('<span class="current">1</span>')
        self.soup = BeautifulSoup(html, "lxml")
        self.get_page_num(self.soup)
        comments = self.get_comment_list(self.soup)
        self.comments.extend(comments)
    def get_page_num(self, soup):
        """获取评论也页数
        """
        page_btns = soup.select(".bottom-page .tcd-number")
        page_total_num = page_btns[-1].string
        self.page_total_num = int(page_total_num)
        print("\033[0;35mPage total num:\033[0m", self.page_total_num)
    def get_comment_list(self, soup):
        comment_list = soup.select(".comment-list .list-item")
        return map(self.parse_comment, comment_list)
    def parse_comment(self, soup):
        name = soup.select_one(".con > .user > .name").string
        href = soup.select_one(".con > .user > .name")['href']
        strings = soup.select_one(".con > .text").strings
        sentences = []
        for string in strings:
            sentences.append(string)
        text = '。'.join(sentences)

        like = soup.select_one(".con > .info > .like span").string
        try:
            like = int(like)
        except TypeError:
            like = 0
        time = soup.select_one(".con > .info > .time").string
        comment = {"name": name, "href": href, "text": text, "like": like, "time": time}
        for value in comment.values():
            assert value is not None
        return comment
    def action(self, num):
        print(f"\033[0;31m{num}\033[0m")
        # self.session.edit_html(".bottom-page", f'<a href="javascript:;" class="tcd-number">{num}</a>')
        self.session.click(".bottom-page .tcd-number", num)
        html = self.session.wait_for_page_loaded(
            f'<span class="current">{num}</span>',
            self._is_wait)
        self.soup = BeautifulSoup(html, "lxml")
        return self.get_comment_list(self.soup)
    def _is_wait(self, html):
        soup = BeautifulSoup(html, "lxml") 
        comment_new = soup.select_one(".comment-list").contents[0]
        comment_old = self.soup.select_one(".comment-list").contents[0]
        return comment_new != comment_old
    def run(self):
        print('start')
        for i in range(2, self.page_total_num+1):
            comments = self.action(i)
            self.comments.extend(comments)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.exit()

if __name__ == "__main__":
    start_time = time.time()
    with Crawler("https://www.bilibili.com/video/BV1zi4y157Jb") as crawler:
        crawler.run()
        # sys.exit(crawler.session.ghost._app.exec_())
        comments = crawler.comments
    print("comment_len:", len(comments))
    frame = pd.DataFrame(comments)
    print(frame.head())
    frame.to_csv(os.path.join(root_path, "comments.csv"), index=False, encoding="utf-8")
    end_time = time.time()
    print("\033[0;35mTime:\033[0m", end_time - start_time)
