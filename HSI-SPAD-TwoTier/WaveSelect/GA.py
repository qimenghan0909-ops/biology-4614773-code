from deap import base #deap用于进化算法的Python框架
from deap import creator
from deap import tools
import numpy as np
import random
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression
# 创建适应度类型和个体类型
# weights=(1.0,) 表示最大化适应度值（如R²）
# 如果是最小化问题（如RMSE），可设置为(-1.0,)
creator.create('FitnessMax', base.Fitness, weights=(1.0,))  # for minimization, set weights as (-1.0,)
# 个体定义为列表，并包含适应度属性
creator.create('Individual', list, fitness=creator.FitnessMax)



def GA(X, y, number_of_generation=10):
    '''
    遗传算法实现特征选择
    :param X: 原始光谱数据，形状为 (样本数, 特征数)
    :param y: 生理指标值，形状为 (样本数,)
    :param number_of_generation: 进化代数，默认10代
    :return: 选中的特征索引数组
    '''
    # 数据标准化：减去均值除以标准差
    scaled_x_train = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1) # 所有特征的最小边界为0
    scaled_y_train = (y - y.mean()) / y.std(ddof=1) # 所有特征的最大边界为1

    # 创建遗传算法工具箱
    toolbox = base.Toolbox()
    # 设置每个特征的选择阈值范围（0-1之间）
    min_boundary = np.zeros(X.shape[1]) # 所有特征的最小边界为0
    max_boundary = np.ones(X.shape[1]) * 1.0  # 所有特征的最大边界为1

    # 所有特征的最大边界为1
    probability_of_crossover = 0.5 # 交叉概率
    probability_of_mutation = 0.2  # 变异概率
    threshold_of_variable_selection = 0.5  #特征选择阈值（大于此值表示选中特征）

    # 定义创建个体的函数（每个基因是0-1之间的随机数）
    def create_ind_uniform(min_boundary, max_boundary):
        index = []
        for min, max in zip(min_boundary, max_boundary):
            index.append(random.uniform(min, max)) # 为每个特征生成随机数
        return index

    # individual 个体
    # population 种群
    # 注册遗传算法组件
    toolbox.register('create_ind', create_ind_uniform, min_boundary, max_boundary) # 个体创建方法
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.create_ind) # 个体初始化
    toolbox.register('population', tools.initRepeat, list, toolbox.individual) #群初始化

    # 定义适应度评估函数
    def evalOneMax(individual):
        individual_array = np.array(individual)
        selected_x_variable_numbers = np.where(individual_array > threshold_of_variable_selection)[0]
        selected_scaled_x_train = scaled_x_train[:, selected_x_variable_numbers]
        # 设置PLS最大主成分数
        max_number_of_components = 10
        # 确定实际可用的主成分数（基于数据秩和最大主成分数）
        if len(selected_x_variable_numbers):
            # cross-validation
            pls_components = np.arange(1, min(np.linalg.matrix_rank(selected_scaled_x_train) + 1,
                                              max_number_of_components + 1), 1)
            # 存储不同主成分数下的R²值
            r2_cv_all = []
            # 遍历所有可能的主成分数
            for pls_component in pls_components:
                # 创建PLS回归模型
                model_in_cv = PLSRegression(n_components=pls_component)
                # 5折交叉验证预测
                estimated_y_train_in_cv = np.ndarray.flatten(
                    model_selection.cross_val_predict(model_in_cv, selected_scaled_x_train, scaled_y_train,
                                                      cv=5))
                # 将预测结果反标准化
                estimated_y_train_in_cv = estimated_y_train_in_cv * y.std(ddof=1) + y.mean()
                # 计算R²（决定系数）
                r2_cv_all.append(
                    1 - sum((y - estimated_y_train_in_cv) ** 2) / sum((y - y.mean()) ** 2))
            # 取所有主成分数中最大的R²作为适应度值
            value = [np.max(r2_cv_all)]
        return value

    # 注册遗传算法操作
    toolbox.register('evaluate', evalOneMax) # 评估函数
    # 加入交叉变换
    toolbox.register('mate', tools.cxTwoPoint) # 两点交叉
    # 设置突变几率
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)  # 位翻转变异（每个基因变异概率5%）
    # 挑选个体
    toolbox.register('select', tools.selTournament, tournsize=3) # 锦标赛选择（3个个体竞争）
    # 种群
    random.seed()
    # 创建初始种群（种群大小等于样本数）
    pop = toolbox.population(n=len(y))

    # 进化循环
    for generation in range(number_of_generation):
        print('-- Generation {0} --'.format(generation + 1))
        # 选择父代个体（锦标赛选择）
        offspring = toolbox.select(pop, len(pop))
        # 克隆选中的个体（避免引用问题）
        offspring = list(map(toolbox.clone, offspring))
        # 交叉操作：对相邻个体进行两点交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < probability_of_crossover:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        # 变异操作
        for mutant in offspring:
            if random.random() < probability_of_mutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # 评估新个体（只评估适应度无效的个体）
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit # 更新适应度值
        # 选出来的个体(描述符)
        print('  Evaluated %i individuals' % len(invalid_ind))
        # 用子代替换父代种群
        pop[:] = offspring
        # 计算种群统计信息
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        # 输出种群统计信息
        print('  Min %s' % min(fits))
        print('  Max %s' % max(fits))
    # 选择最佳个体
    best_individual = tools.selBest(pop, 1)[0]
    best_individual_array = np.array(best_individual)
    # 从最佳个体中提取选中的特征索引
    selected_x_variable_numbers = np.where(best_individual_array > threshold_of_variable_selection)[0]

    return selected_x_variable_numbers
