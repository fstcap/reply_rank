from threading import Thread, currentThread
from queue import Queue
from contextlib import contextmanager 

StopEvent = object()

class ThreadPool:
    def __init__(self, max_num, max_task_num=None):
        """
        初始化线程池
        :param max_num: 线程池最大线程数量
        :param max_task_num: 任务队列长度
        """
        if max_task_num:
            self.q = Queue(max_task_num)
        else:
            self.q = Queue()
        self.max_num = max_num
        
        self.cancel = False
        self.terminal = False

        self.generate_list = []
        self.free_list = []
    def put(self, func, args, callback=None):
        """
        往任务队列里放入一个任务
        :param func: 任务函数
        :param args: 任务函数所需参数
        :param callback: 任务执行失败或成功后执行的回调函数，回调函数有两个参数
        1、任务函数执行状态；2、任务函数返回值（默认为None，即：不执行回调函数）
        :return: 如果线程池已经终止，则返回True否则None
        """
        if self.cancel:
            return
        if len(self.free_list) == 0 and len(self.generate_list) < self.max_num:
            self.generate_thread()
        w = (func, args, callback,)
        self.q.put(w)
    def generate_thread(self):
        """
        创建一个线程
        """
        t = Thread(target=self.call)
        t.start()
    def call(self):
        """
        循环去获取任务函数并执行任务函数。在正常情况下，每个线程都保存生存状态，  直到获取线程终止的flag。
        """
        current_thread = currentThread().getName()
        self.generate_list.append(current_thread)
        event = self.q.get()
        while event != StopEvent:
            func, arguments, callback = event
            try:
                result = func(current_thread, *arguments)
                success = True
            except Exception as e:
                result = None
                success = False
            if callback is not None:
                try:
                    callback(success, result)
                except Exception as e:
                    pass
            with self.worker_state(self.free_list, current_thread):
                if self.terminal:
                    event = StopEvent
                else:
                    event = self.q.get()
        else:
            self.generate_list.remove(current_thread)
    def close(self):
        """
        执行完所有的任务后，让所有线程都停止的方法
        """
        self.cancel = True
        full_size = len(self.generate_list)
        while full_size:
            self.q.put(StopEvent)
            full_size -=1
    def terminate(self):
        """
        在任务执行过程中，终止线程，提前退出。
        """
        self.terminal = True
        while self.generate_list:
            self.q.put(StopEvent)
    @contextmanager
    def worker_state(self, state_list, worker_thread):
        """
        用于记录空闲的线程，或从空闲列表中取出线程处理任务
        """
        state_list.append(worker_thread)
        try:
            yield
        finally:
            state_list.remove(worker_thread)
