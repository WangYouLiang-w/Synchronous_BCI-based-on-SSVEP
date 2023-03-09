import keyboard
import threading
import queue
import time

'''按键测试'''
class test(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def keydownFunc(self,x):
        # if keyboard.is_pressed('w'):
        #     print('w')
        w = keyboard.KeyboardEvent('down', 28, 'w')
        if x.event_type == 'down' and x.name == w.name:
            print('w')
        else:
            print('!!')
        print(1)

    def keydown_release(self,x):
        w = keyboard.KeyboardEvent('up', 28, 'w')
        if x.event_type == 'up' and x.name == w.name:
            print('!!')

    def run(self):
        # while True:
        #     print(len(keyboard.read_key()))
        #     if len(keyboard.read_key()):
        #         self.keydownFunc()
        #     self.keydown_release()
        while True:
            keyboard.hook(self.keydownFunc)
            #
            # keyboard.hook(self.keydown_release)
            keyboard.wait()


if __name__ == '__main__':
    a = test()
    a.setDaemon(True)
    a.start()

    while True:
        # print(1)
        time.sleep(0.005)





'''队列测试'''
# import queue,threading,time
# q=queue.Queue(5)

# put_nowait（非阻塞模式） 如果队列满了，会抛出异常  put（阻塞模式），如果队列满了就会发生阻塞
# get_nowait（非阻塞模式） 如果队列空了，会抛出异常  get（阻塞模式），如果队列空了就会发生阻塞
# q.put(1)
# q.put(1)
# q.put(1)
# q.put(1)
# q.put(1)

# print(q.get())
# print(q.get())
# print(q.get())
# print(q.get())
# print(q.get())
# print(q.get())

''' 队列可实现生产者和消费者解耦有利于解决高并发问题'''
# def product(arg):
#     q.put(str(arg)+'包子')
# def consumer(arg):
#     print(arg,q.get())
# for i in range(3):
#     t=threading.Thread(target=product,args=(i,))
#     t.start()
# for j in range(3):
#     t=threading.Thread(target=consumer,args=(j,))
#     t.start()
