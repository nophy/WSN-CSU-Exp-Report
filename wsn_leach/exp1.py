import numpy as np
import matplotlib.pyplot as plt
import turtle

def run():
    """
    1、输入节点个数N
    2、node_factory(N): 生成N个节点的列表
    3、classify(nodes,flag,k=10): 进行k轮簇分类，flag已标记的节点不再成为簇首，返回所有簇的列表
    4、show_plt(classes): 迭代每次聚类结果，显示连线图
    :return:
    """
    # 节点数目和圆形网络半径
    N = 500
    R = 10
    r = 4
    # 获取初始节点列表
    top, iso = node_factory(N, R, r)
    
    # 显示拓扑结构
    show_plt(top, iso, R)

    # 保存拓扑结构为txt文件
    save_top(top, iso)

def dist(v_A, v_B):
    """
    判断两个节点之间的欧几里得距离
    :param v_A: A 二维向量
    :param v_B: B 二维向量
    :return: 一维距离
    """
    return np.sqrt(np.power((v_A[0] - v_B[0]), 2) + np.power((v_A[1] - v_B[1]), 2))

def node_factory(N, R, r):
    """
    生成N个节点的集合
    :param N: 节点的数目
    :param R: 圆形拓扑半径
    :param r: 节点通信半径
    :param nodes: 节点的集合
    :param iso: 标志:是否为孤立节点
    :return: 节点集合nodes=[[x,y],[x,y]...] + 标志iso
    """
    nodes = []
    iso = []

    sinknode = [0, 0]
    nodes.append(sinknode)
    iso.append(0)
    
    i = 0
    while i < N-1:
        # 在1*1矩阵生成[x,y]坐标，并根据离sink节点的距离做判断是否为孤立节点
        node = [np.random.uniform(-1, 1)*R, np.random.uniform(-1, 1)*R]
        if dist(node, sinknode) < R and dist(node, sinknode) > r:
            nodes.append(node)
            iso.append(1)
            i = i + 1
        elif dist(node, sinknode) < R and dist(node, sinknode) < r:
            nodes.append(node)
            iso.append(0)
            i = i + 1


    return nodes, iso


def show_plt(top, iso, R):
    """
    显示分类图
    :param top: [[类1...],[类2...]....]-->[簇首,成员,成员...]
    :return:
    """
    fig = plt.figure()
    ax1 = plt.gca()
 
    # 设置标题
    ax1.set_title('WSN1')
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')

    for i in range(len(top)):
        plt.scatter(top[i][0], top[i][1], color='b')
        if i != 0 and iso[i] != 1:
            plt.plot([top[0][0], top[i][0]], [top[0][1], top[i][1]], color='r')

    a, b = (top[0][0], top[0][1])
    theta = np.arange(0, 2*np.pi, 0.01)
    x = a + R * np.cos(theta)
    y = b + R * np.sin(theta)
    plt.plot(x, y, 'y')
    plt.show()

def save_top(top, iso):
    """
    :param top: 网络拓扑结构
    :param iso: 是否为孤立节点标志
    """
    np.savetxt('top.txt', top, fmt='%.5f', delimiter=',')
    np.savetxt('iso.txt', iso, fmt='%d', delimiter=',')

if __name__ == '__main__':
    run()
