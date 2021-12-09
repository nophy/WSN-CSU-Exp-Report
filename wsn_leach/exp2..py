# from typing_extensions import ParamSpec
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

 
def dist(v_A, v_B):
    """
    判断两个节点之间的一维距离
    :param v_A: A 二维向量
    :param v_B: B 二维向量
    :return: 一维距离
    """
    return np.sqrt(np.power((v_A[0] - v_B[0]), 2) + np.power((v_A[1] - v_B[1]), 2))
 

def node_factory(N, R, r, energy=50):
    """
    生成N个节点的拓扑网络
    :param N: 网络中节点个数
    :param R: 圆形拓扑半径
    :para r: 通信半径范围，超出此范围为某簇的孤立节点
    :param selected_flag: 标志:是否被选择为簇首-->初始化为0
    :param energy: 能量
    :return: 节点集合nodes=[[x,y,e],[x,y,e]...]
    """
    nodes = []
    selected_flag = []
    iso = []

    # 中心sink节点
    sinknode = [0, 0, energy]
    nodes.append(sinknode)
    selected_flag.append(0)
    
    # 随机生成圆形拓扑网络
    i = 0
    while i < N-1:
        # 在1*1矩阵生成[x,y]坐标
        node = [np.random.uniform(-1, 1)*R, np.random.uniform(-1, 1)*R, energy]
        if dist(node, sinknode) < R and dist(node, sinknode) > r:
            nodes.append(node)
            selected_flag.append(0)
            iso.append(1)
            i = i + 1
        elif dist(node, sinknode) < R and dist(node, sinknode) < r:
            nodes.append(node)
            selected_flag.append(0)
            iso.append(0)
            i = i + 1

            
    return nodes, selected_flag, iso
 
 
def sel_heads(r, nodes, flags):
    """
    使用leach协议，选取簇头（注意这里还没有开始进入正式的分簇，这里只选了簇头）
    :param r: 轮数
    :param nodes: 节点列表
    :param flags: 选择标志
    :param P: 比例因子
    :return: 簇首列表heads,簇成员列表members
    """
    # 阈值函数 Tn 使用leach计算
    P = 0.25 * (200 / len(nodes))
    Tn = P / (1 - P * (r % (1 / P)))
    heads = []  # 簇首列表
    members = []    # 簇成员列表
    n_head = 0  # 本轮簇首数
    rands = [np.random.random() for _ in range(len(nodes))] # 对每个节点生成对应的随机数，用于筛选簇头
 
    # 遍历随机数列表，选取簇首
    for i in range(len(nodes)):
        # 随机数低于阈值-->选为簇首
        if rands[i] <= Tn:
            flags[i] = 1
            heads.append(nodes[i])
            n_head += 1
        # 随机数高于阈值
        else:
            members.append(nodes[i])
 
    return heads, members
 
 
def classify(nodes, flag, r, mode=1, k=20):
    """
    对网络进行簇分类
    :param nodes: 节点列表
    :param flag: 节点标记
    :param mode: 0-->显示图片(死亡节点不显示)  1-->显示结束轮数
    :param k: 轮数
    :return: 簇分类结果列表 classes[[类1..],[类2...],......]  [类1...簇首...簇成员]
    """
    # 能量损耗模型的参数
    b = 2400    # 比特数
    e_elec = 5*np.power(10., -9)*100
    e_fs = 10*np.power(10., -12)*100
    e_mp = 0.0013*np.power(10., -12)*100
    d_0 = 4.0   # 阈值

    # k轮的有效集合: 无死亡节点
    iter_classes = []
    # 是否已有节点能量为0
    e_is_empty = 0
 
    # 迭代r轮
    for r in range(k):
        # mode1: 若无死亡节点 继续迭代
        if e_is_empty == 0:
            # 获取簇首列表，簇成员列表
            heads, members = sel_heads(r,nodes,flag)
            
            # 建立簇类的列表
            if len(heads) == 0:
                break
            classes = [[] for _ in range(len(heads))]
 
            # 将簇首作为首节点添加到聚类列表中
            for i in range(len(heads)):
                classes[i].append(heads[i])
 
            # 簇分类:遍历节点node
            for member in members:
 
                # 选取距离最小的节点
                dist_min = 100000
 
                # 判断和每个簇首的距离
                for i in range(len(heads)):
 
                    dist_heads = dist(member, heads[i])
 
                    # 找到距离最小的簇头对应的heads下标i
                    if dist_heads < dist_min:
                        dist_min = dist_heads
                        head_cla = i
                if dist_min==1:
                    print("本轮没有簇首!")
                    break
                    # 添加到距离最小的簇首对应的聚类列表中

                classes[head_cla].append(member)
                
                # 正式的数据传输过程，使用能量消耗模型
                if int(member[2]) > 0 and int(heads[head_cla][2]) > 0:
                    if dist_min < d_0:
                        member[2] -= e_elec*b+e_fs*b*dist_min
                    else:
                        member[2] -= e_elec*b+e_mp*b*dist_min
                    
                    if dist([member[0], member[1]], [heads[head_cla][0], heads[head_cla][1]]) <= r:
                        heads[head_cla][2] -= e_elec*b
                    else:
                        pass
                    # heads[head_cla][2] -= e_elec*b
                else:
                    e_is_empty = mode
                    # break
            iter_classes.append(classes)
 
        else:
            print("第", r, "轮能量耗尽")
            break
 
    return iter_classes
 
 
def show_plt(classes, R, r):
    """
    显示分类图
    :param classes: [[类1...],[类2...]....]-->[簇首,成员,成员...]
    :param R: 圆形拓扑半径
    :para r: 通信半径范围，超出此范围为某簇的孤立节点
    :return:
    """
    fig = plt.figure()
    ax1 = plt.gca()
 
    # 设置标题
    ax1.set_title('WSN2')
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
 
    # 簇内的显示点图标及连线颜色，以得到较好的显示结果
    icon = ['o', '*', '.', 'x', '+', 's']
    color = ['b', 'g', 'c', 'y', 'm']
    
    x, y, e = [], [], []

    # 对不同的簇进行不同的显示，以得到较好的显示结果
    for i in range(len(classes)):
        centor = classes[i][0]
        x.append(centor[0])
        y.append(centor[1])
        e.append(centor[2])
        for point in classes[i]:
            if point[2] > 0 and dist(centor, point) < r:
                ax1.plot([centor[0], point[0]], [centor[1], point[1]], c=color[i % 5], marker=icon[i % 5], alpha=0.4)
            elif point[2] > 0 and dist(centor, point) > r:
                ax1.plot([centor[0], point[0]], [centor[1], point[1]], c='r', marker=icon[i % 5], alpha=0.4)
            else:
                pass


    a, b = (0., 0.)
    theta = np.arange(0, 2*np.pi, 0.01)
    x = a + R * np.cos(theta)
    y = b + R * np.sin(theta)
    plt.plot(x, y, 'y')

    # 显示
    plt.show()
 
def show_eninfo(iter_classes):
    fig = plt.figure()
    ax1 = Axes3D(fig)
    lastclass = iter_classes[-1]

    x, y, e = [], [], []

    # 将所有节点的剩余能量统计起来，用于后续能量三维图的显示
    for i in range(len(lastclass)):
        for point in lastclass[i]:
            x.append(point[0])
            y.append(point[1])
            e.append(point[2])

    # 需要进行数据类型转换list->ndarray，才能进行三维图像的显示
    x = np.array(x)
    y = np.array(y)
    e = np.array(e)

    # 显示三维图像
    print(f"各节点能量的标准差: {np.std(e):.4f}")
    ax1.plot_trisurf(x, y, e, cmap='rainbow')
    plt.show()

def run():
    """
    1、输入节点个数N
    2、node_factory(N,energy): 生成N个节点的列表，节点的能量初始化为energy
    3、classify(nodes,mode=1,k=100): 进行簇分类，返回所有簇的列表
       mode=0: 当节点死亡不停止，进行k次迭代，显示k张图，图中已死亡节点不标记
       mode=1: 当节点死亡停止，记录第一个节点死亡时的轮数，显示无死亡节点的图
    4、show_plt(classes): 迭代每次聚类结果，显示连线图
    :return:
    """
    # N = int(input("请输入节点个数:"))
    N = 300
    R = 20
    r = 5

    # 获取初始节点列表
    nodes, flag, iso = node_factory(N, R, r, energy=5)
    # 对节点列表进行簇分类,mode为模式 2种
    iter_classes = classify(nodes,flag, r, mode=1, k=200)
    # 迭代每次聚类结果，显示连线图
    for classes in iter_classes:
        # 显示分类结果
        show_plt(classes, R, r)

    show_eninfo(iter_classes)
if __name__ == '__main__':
    run()