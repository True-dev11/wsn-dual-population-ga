# -*- coding: utf-8 -*-
"""
无线传感器网络节点部署优化
- 普通GA（基线）
- 双种群GA（改进算法）
Windows 11 + Python 3.x
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

# ============ 中文显示 ============
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ 问题参数 ============
L = 100.0                 # 区域长
W = 100.0                 # 区域宽
N = 25                    # 节点数量
r_s = 15.0                # 感知半径
sink = np.array([0.0, 0.0])   # 边缘服务器坐标
GRID = 50                 # 覆盖率网格分辨率

w1 = 0.7                  # 覆盖率权重
w2 = 0.3                  # 能耗权重
alpha = 1.0               # 能耗系数
D = 1.0                   # 每节点数据量

d_max = np.sqrt(L**2 + W**2)
E_ref = N * alpha * D * d_max**2

# ============ 覆盖率 ============
def coverage(xs, ys):
    gx = np.linspace(0, L, GRID)
    gy = np.linspace(0, W, GRID)
    GX, GY = np.meshgrid(gx, gy)
    covered = np.zeros_like(GX, dtype=bool)
    for xi, yi in zip(xs, ys):
        covered |= ((GX - xi)**2 + (GY - yi)**2) <= r_s**2
    return covered.mean()

# ============ 能耗 ============
def energy(xs, ys):
    d2 = (xs - sink[0])**2 + (ys - sink[1])**2
    return alpha * D * np.sum(d2)

# ============ 目标函数 ============
def objective(ind):
    xs = ind[0::2]
    ys = ind[1::2]
    C = coverage(xs, ys)
    E = energy(xs, ys)
    E_norm = E / E_ref
    F = w1 * (1 - C) + w2 * E_norm
    return F

def fitness(ind):
    return 1.0 / (objective(ind) + 1e-8)

# ============ 初始化 ============
def random_individual(mode='global'):
    if mode == 'global':
        xs = np.random.uniform(0, L, N)
        ys = np.random.uniform(0, W, N)
    elif mode == 'sink':
        # 靠近边缘服务器初始化
        xs = np.random.uniform(0, L*0.5, N)
        ys = np.random.uniform(0, W*0.5, N)
    ind = np.empty(2*N)
    ind[0::2] = xs
    ind[1::2] = ys
    return ind

def init_population(size, mode='global'):
    return np.array([random_individual(mode) for _ in range(size)])

# ============ 锦标赛选择 ============
def tournament_selection(pop, fits, k=3):
    idx = np.random.choice(len(pop), k, replace=False)
    best = idx[np.argmax(fits[idx])]
    return pop[best].copy()

# ============ 多点交叉 ============
def multi_point_crossover(p1, p2, pc=0.8):
    if np.random.rand() > pc:
        return p1.copy(), p2.copy()
    n = len(p1)
    points = sorted(np.random.choice(n, 3, replace=False))
    c1, c2 = p1.copy(), p2.copy()
    c1[points[0]:points[1]] = p2[points[0]:points[1]]
    c2[points[0]:points[1]] = p1[points[0]:points[1]]
    c1[points[2]:] = p2[points[2]:]
    c2[points[2]:] = p1[points[2]:]
    return c1, c2

# ============ 高斯变异 ============
def gaussian_mutation(ind, pm=0.1, sigma=10.0):
    ind = ind.copy()
    for i in range(len(ind)):
        if np.random.rand() < pm:
            ind[i] += np.random.normal(0, sigma)
    # 边界截断
    ind[0::2] = np.clip(ind[0::2], 0, L)
    ind[1::2] = np.clip(ind[1::2], 0, W)
    return ind

# ============ 普通GA ============
def standard_ga(pop_size=50, generations=100, pc=0.8, pm=0.1, elite=2, seed=0):
    np.random.seed(seed)
    pop = init_population(pop_size, 'global')
    fits = np.array([fitness(ind) for ind in pop])
    best_hist = []

    for g in range(generations):
        new_pop = []
        # 精英
        elite_idx = np.argsort(-fits)[:elite]
        for i in elite_idx:
            new_pop.append(pop[i].copy())
        # 生成其余
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fits)
            p2 = tournament_selection(pop, fits)
            c1, c2 = multi_point_crossover(p1, p2, pc)
            c1 = gaussian_mutation(c1, pm)
            c2 = gaussian_mutation(c2, pm)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        pop = np.array(new_pop)
        fits = np.array([fitness(ind) for ind in pop])
        best_hist.append(1.0 / fits.max())

    best_idx = np.argmax(fits)
    diversity = np.std(pop)
    return pop[best_idx], 1.0 / fits[best_idx], best_hist, diversity

# ============ 双种群GA ============
def dual_population_ga(pop_size=50, generations=100, pc=0.8, pm=0.1,
                       elite=2, migrate_interval=10, migrate_ratio=0.1, seed=0):
    np.random.seed(seed)
    # 种群A：全局探索
    popA = init_population(pop_size, 'global')
    # 种群B：靠近sink开发
    popB = init_population(pop_size, 'sink')

    fitsA = np.array([fitness(ind) for ind in popA])
    fitsB = np.array([fitness(ind) for ind in popB])
    best_hist = []

    migrate_num = max(1, int(pop_size * migrate_ratio))

    def evolve(pop, fits, mode):
        new_pop = []
        elite_idx = np.argsort(-fits)[:elite]
        for i in elite_idx:
            new_pop.append(pop[i].copy())
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fits)
            p2 = tournament_selection(pop, fits)
            c1, c2 = multi_point_crossover(p1, p2, pc)
            sig = 15.0 if mode == 'A' else 5.0
            c1 = gaussian_mutation(c1, pm, sig)
            c2 = gaussian_mutation(c2, pm, sig)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        return np.array(new_pop)

    for g in range(generations):
        popA = evolve(popA, fitsA, 'A')
        popB = evolve(popB, fitsB, 'B')
        fitsA = np.array([fitness(ind) for ind in popA])
        fitsB = np.array([fitness(ind) for ind in popB])

        # 迁移
        if (g + 1) % migrate_interval == 0:
            idxA = np.argsort(-fitsA)[:migrate_num]
            idxB = np.argsort(-fitsB)[:migrate_num]
            # A -> B
            worstB = np.argsort(fitsB)[:migrate_num]
            for i, j in zip(worstB, idxA):
                popB[i] = popA[j].copy()
            # B -> A
            worstA = np.argsort(fitsA)[:migrate_num]
            for i, j in zip(worstA, idxB):
                popA[i] = popB[j].copy()
            fitsA = np.array([fitness(ind) for ind in popA])
            fitsB = np.array([fitness(ind) for ind in popB])

        best_fit = max(fitsA.max(), fitsB.max())
        best_hist.append(1.0 / best_fit)

    all_pop = np.vstack([popA, popB])
    all_fits = np.array([fitness(ind) for ind in all_pop])
    best_idx = np.argmax(all_fits)
    diversity = np.std(all_pop)
    return all_pop[best_idx], 1.0 / all_fits[best_idx], best_hist, diversity

# ============ 画图 ============
def plot_deployment(ind, title):
    xs = ind[0::2]
    ys = ind[1::2]
    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, c='red', s=40, label='传感器节点')
    for xi, yi in zip(xs, ys):
        circle = plt.Circle((xi, yi), r_s, color='blue', alpha=0.08)
        plt.gca().add_patch(circle)
    plt.scatter([sink[0]], [sink[1]], c='green', marker='*', s=200, label='边缘服务器')
    plt.xlim(-5, L+5)
    plt.ylim(-5, W+5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(title)
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

# ============ 统计工具 ============
def compute_diversity(pop):
    """末代多样性：所有基因的标准差"""
    return np.std(pop)

def run_experiments(runs=20):
    results = {
        'ga':   {'F': [], 'time': [], 'conv': [], 'div': []},
        'dual': {'F': [], 'time': [], 'conv': [], 'div': []},
    }
    hist_ga_all = []
    hist_dual_all = []
    best_ga = None
    best_dual = None

    for r in range(runs):
        seed = r + 1
        print(f"---- 第 {r+1}/{runs} 次运行 (seed={seed}) ----")

        # 普通GA
        t1 = time.time()
        best1, f1, hist1, div1 = standard_ga(pop_size=100, seed=seed)
        t1 = time.time() - t1
        results['ga']['F'].append(f1)
        results['ga']['time'].append(t1)
        results['ga']['conv'].append(len(hist1))
        results['ga']['div'].append(div1)
        hist_ga_all.append(hist1)
        if best_ga is None or f1 < best_ga[1]:
            best_ga = (best1, f1)

        # 双种群GA
        t2 = time.time()
        best2, f2, hist2, div2 = dual_population_ga(pop_size=50, seed=seed)
        t2 = time.time() - t2
        results['dual']['F'].append(f2)
        results['dual']['time'].append(t2)
        results['dual']['conv'].append(len(hist2))
        results['dual']['div'].append(div2)
        hist_dual_all.append(hist2)
        if best_dual is None or f2 < best_dual[1]:
            best_dual = (best2, f2)

        print(f"  普通GA   F={f1:.4f}  用时={t1:.2f}s")
        print(f"  双种群GA F={f2:.4f}  用时={t2:.2f}s")

    return results, hist_ga_all, hist_dual_all, best_ga, best_dual


def summarize(results):
    print("\n================ 统计结果 ================")
    header = f"{'指标':<12}{'普通GA':>15}{'双种群GA':>15}"
    print(header)
    print("-" * len(header))

    def line(name, fn, fmt='.4f'):
        v1 = fn(results['ga']['F'])
        v2 = fn(results['dual']['F'])
        print(f"{name:<12}{v1:>15{fmt}}{v2:>15{fmt}}")

    line('最优值', min)
    line('最差值', max)
    line('平均值', np.mean)
    line('方差', np.var, fmt='.4e')

    t1 = np.mean(results['ga']['time'])
    t2 = np.mean(results['dual']['time'])
    print(f"{'平均运行时间':<12}{t1:>15.4f}{t2:>15.4f}")

    # 平均收敛代数
    conv1 = np.mean(results['ga']['conv'])
    conv2 = np.mean(results['dual']['conv'])
    print(f"{'平均收敛代数':<12}{conv1:>15.2f}{conv2:>15.2f}")

    # 成功率：达到最优±1%
    best1 = min(results['ga']['F'])
    best2 = min(results['dual']['F'])
    thr1 = best1 * 1.01
    thr2 = best2 * 1.01
    suc1 = np.mean(np.array(results['ga']['F']) <= thr1)
    suc2 = np.mean(np.array(results['dual']['F']) <= thr2)
    print(f"{'成功率':<12}{suc1:>15.4f}{suc2:>15.4f}")

    # 末代多样性
    div1 = np.mean(results['ga']['div']) if results['ga']['div'] else 0
    div2 = np.mean(results['dual']['div']) if results['dual']['div'] else 0
    print(f"{'末代多样性':<12}{div1:>15.4f}{div2:>15.4f}")

    # 约束满足率：本问题通过边界截断保证，故均为 1
    print(f"{'约束满足率':<12}{1.0:>15.4f}{1.0:>15.4f}")
    print("=========================================\n")


# ============ 主程序 ============
if __name__ == "__main__":
    results, hist_ga_all, hist_dual_all, best_ga, best_dual = run_experiments(runs=20)

    summarize(results)

    # 平均收敛曲线
    min_len = min(min(len(h) for h in hist_ga_all),
                  min(len(h) for h in hist_dual_all))
    ga_mat = np.array([h[:min_len] for h in hist_ga_all])
    dual_mat = np.array([h[:min_len] for h in hist_dual_all])
    ga_mean = ga_mat.mean(axis=0)
    dual_mean = dual_mat.mean(axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(ga_mean, label='普通GA 平均', linewidth=2)
    plt.plot(dual_mean, label='双种群GA 平均', linewidth=2)
    plt.xlabel('迭代代数')
    plt.ylabel('最优目标函数值 F')
    plt.title('20 次独立运行平均收敛曲线')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('convergence_avg.png', dpi=150)
    plt.show()

    # 最优部署图
    plot_deployment(best_ga[0], '普通GA 最优部署')
    plot_deployment(best_dual[0], '双种群GA 最优部署')