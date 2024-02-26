import os

import dill
# import networkx as nx
import pandas as pd
# import statsmodels.api as sm
# from cdt.causality.graph import GES
# from dowhy import CausalModel
from tqdm import tqdm


# 因果图构建
class CausaltyGraph4Visit:
    def __init__(self, data_all, data_train, num_diagnosis, num_procedure, num_medication, dataset):
        """
        data_all是全局数据
        data_train是训练集中的数据
        剩下三个是不同种实体的个数
        """
        self.dataset = dataset

        # diag,proc,med的数量
        self.num_d = num_diagnosis
        self.num_p = num_procedure
        self.num_m = num_medication

        # 从训练集中产生的df大表，标志着每个实体的出现
        self.data = self.data_process(data_train)

        # 用所有数据生成的同构图，不包含药物与疾病的关系，只包含"d-d","p-p","m-m"三种数据，不参与训练，只参与推理（因为包含了测试集样本）
        self.causal_graphs = self.build_graph(data_all)

        # 通过self.data（训练集出的图）而产生的因果效应
        self.diag_med_effect = self.build_effect(num_diagnosis, num_medication, "Diag", "Med")
        self.proc_med_effect = self.build_effect(num_procedure, num_medication, "Proc", "Med")

    # 返回就诊中的一个子图
    def get_graph(self, graph_id, graph_type):
        graph = self.causal_graphs[graph_id]

        if graph_type == "Diag":
            return graph[0]
        elif graph_type == "Proc":
            return graph[1]
        elif graph_type == "Med":
            return graph[2]

    # 返回任意两个疾病-药物之间的因果关系
    def get_effect(self, a, b, A_type, B_type):
        a = A_type + '_' + str(int(a))
        b = B_type + '_' + str(int(b))

        if A_type == "Diag" and B_type == "Med":
            effect_df = self.diag_med_effect
        elif A_type == "Proc" and B_type == "Med":
            effect_df = self.proc_med_effect
        else:
            raise ValueError("Invalid A_type and B_type combination")

        effect = effect_df.loc[a, b]
        return effect

    def get_threshold_effect(self, threshold, A_type, B_type):
        if A_type == "Diag" and B_type == "Med":
            effect_df = self.diag_med_effect
        elif A_type == "Proc" and B_type == "Med":
            effect_df = self.proc_med_effect
        else:
            raise ValueError("Invalid A_type and B_type combination")

        # 将 DataFrame 转换为一维序列
        flattened = effect_df.stack()

        # 计算并返回对应的阈值
        threshold_value = flattened.quantile(threshold)
        return threshold_value

    # 建立全部的因果关系
    def build_effect(self, num_a, num_b, a_type, b_type):
        # 检查本地是否有保存的结果
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, f"../../data/{self.dataset}/graphs/{a_type}_{b_type}_causal_effect.pkl")
        # 构建完整的文件路径
        try:
            effect_df = dill.load(open(file_path, "rb"))
        except FileNotFoundError:
            print(f"你的本地没有关于的基于无图的因果效应，正在建立中，这大概需要几个小时的时间..")
            processed_data = self.data

            effect_df = pd.DataFrame(0.0, index=[f"{a_type}_{i}" for i in range(num_a)],
                                     columns=[f"{b_type}_{j}" for j in range(num_b)])

            for i in tqdm(range(num_a)):
                for j in range(num_b):
                    causal_value = self.compute_causal_value(processed_data, i, j, a_type, b_type)
                    effect_df.at[f"{a_type}_{i}", f"{b_type}_{j}"] = causal_value
                    print(f"{a_type}:{i}, {b_type}:{j}, causal_value:{causal_value}")

            # 保存为 NumPy 数组
            with open(file_path, "wb") as f:
                dill.dump(effect_df, f)

        return effect_df

    # 因果效应推断（不基于图，基于图会有信    息泄露）
    # def compute_causal_value(self, data, d, m, a_type, b_type):
    #     selected_data = data[[f'{a_type}_{d}', f'{b_type}_{m}']]
    #     model = CausalModel(data=selected_data, treatment=f'{a_type}_{d}', outcome=f'{b_type}_{m}')
    #     # 估计因果效应
    #     identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    #     estimate = model.estimate_effect(identified_estimand,
    #                                      method_name="backdoor.generalized_linear_model",
    #                                      method_params={"glm_family": sm.families.Binomial()})
    #     return estimate.value

    # 为每个会话构建三种图"d-d","p-p","m-m"
    def build_graph(self, data_all):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建完整的文件路径
        file_path = os.path.join(current_dir, f"../../data/{self.dataset}/graphs/causal_graph.pkl")
        try:
            subgraph_list = dill.load(open(file_path, "rb"))
        except FileNotFoundError:
            # causal_graphs = []
            # # 生成因果图的部分抹掉了，为了能不用cdt
            # print("构建全部因果图..")
            # sessions = self.sessions_process(data_all)
            # for adm in tqdm(sessions):
            #     D = adm[0]
            #     P = adm[1]
            #     M = adm[2]
            #     # 将数据转化为DataFrame列名
            #     visit = [f"Diag_{d}" for d in D] + [f"Proc_{p}" for p in P] + [f"Med_{m}" for m in M]
            #     visit_data = self.data[visit]
            #     # 采用GES的算法计算因果图
            #     cdt_algo = GES()
            #     causal_graph = cdt_algo.predict(visit_data)
            #     # 去掉Med-Diag，Med-Proc
            #     new_graph = nx.DiGraph()
            #     # 首先，将所有节点添加到新图中
            #     for node in causal_graph.nodes():
            #         new_graph.add_node(node)
            #     # 然后，将需要的边添加到新图中
            #     for edge in causal_graph.edges():
            #         source, target = edge
            #         # 保留 病-药，病-病，药-药
            #         if source.startswith("Diag") and target.startswith("Diag"):
            #             new_graph.add_edge(source, target)
            #         elif source.startswith("Diag") and target.startswith("Med"):
            #             new_graph.add_edge(source, target)
            #         elif source.startswith("Diag") and target.startswith("Proc"):
            #             new_graph.add_edge(source, target)
            #         elif source.startswith("Proc") and target.startswith("Proc"):
            #             new_graph.add_edge(source, target)
            #         elif source.startswith("Proc") and target.startswith("Diag"):
            #             new_graph.add_edge(source, target)
            #         elif source.startswith("Proc") and target.startswith("Med"):
            #             new_graph.add_edge(source, target)
            #         elif source.startswith("Med") and target.startswith("Med"):
            #             new_graph.add_edge(source, target)
            #
            #     causal_graph = new_graph
            #
            #     # 4.移除环路
            #     while not nx.is_directed_acyclic_graph(causal_graph):
            #         cycle_nodes = nx.find_cycle(causal_graph, orientation="original")
            #
            #         for edge in cycle_nodes:
            #             source, target, _ = edge
            #             causal_graph.remove_edge(source, target)
            #         # print(f"disease{d} - medication{m}之中我们无法捕捉到合格的DAG因果图，我们试着移除环路！"
            #         #       f"还剩下{causal_graph.edges()}")
            #     causal_graph = nx.DiGraph(causal_graph)
            #     causal_graphs.append(causal_graph)

            subgraph_list = []
            # for graph in tqdm(causal_graphs):
            #     graph_type = []
            #
            #     nodes_to_remove = [node for node in graph.nodes() if "Med" in node or "Proc" in node]
            #     graph2 = graph.copy()
            #     graph2.remove_nodes_from(nodes_to_remove)
            #     graph_type.append(graph2)
            #
            #     nodes_to_remove = [node for node in graph.nodes() if "Diag" in node or "Med" in node]
            #     graph2 = graph.copy()
            #     graph2.remove_nodes_from(nodes_to_remove)
            #     graph_type.append(graph2)
            #
            #     nodes_to_remove = [node for node in graph.nodes() if "Diag" in node or "Proc" in node]
            #     graph2 = graph.copy()
            #     graph2.remove_nodes_from(nodes_to_remove)
            #     graph_type.append(graph2)
            #
            #     subgraph_list.append(graph_type)
            #
            # dill.dump(subgraph_list, open(file_path, "wb"))
        return subgraph_list

    # 将所有就诊变成会话形式，一条儿一条儿的
    def sessions_process(self, raw_data):
        sessions = []
        for patient in raw_data:
            for adm in patient:
                sessions.append(adm)
        return sessions

    # 将一条一条的会话形式数据变成一个df大表
    def data_process(self, data_train):
        # 获取当前脚本文件所在的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建完整的文件路径
        file_path = os.path.join(current_dir, f'../../data/{self.dataset}/graphs/matrix4causalgraph.pkl')
        try:
            with open(file_path, "rb") as f:
                df = dill.load(f)
        except FileNotFoundError:

            print("整理数据集..")
            train_sessions = self.sessions_process(data_train)

            df = pd.DataFrame(0.0, index=range(len(train_sessions)), columns=
            [f'Diag_{i}' for i in range(self.num_d)] +
            [f'Proc_{i}' for i in range(self.num_p)] +
            [f'Med_{i}' for i in range(self.num_m)])

            for i, session in tqdm(enumerate(train_sessions)):
                D, P, M, _ = session
                df.loc[i, [f'Diag_{d}' for d in D]] = 1
                df.loc[i, [f'Proc_{p}' for p in P]] = 1
                df.loc[i, [f'Med_{m}' for m in M]] = 1

            with open(file_path, "wb") as f:
                dill.dump(df, f)
        return df


if __name__ == '__main__':
    # 测试样例
    data_path = "../../data/mimic3/output/records_final.pkl"
    voc_path = "../../data/mimic3/output/voc_final.pkl"
    data = dill.load(open(data_path, "rb"))

    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    adm_id = 0
    for patient in data:
        for adm in patient:
            adm.append(adm_id)
            adm_id += 1
    data = data[:5]

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    causal_graph = CausaltyGraph4Visit(data, data_train, voc_size[0], voc_size[1], voc_size[2], 'mimic3')
