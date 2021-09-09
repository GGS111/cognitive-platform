from matplotlib import image as mpimg
import pickle
import numpy as np
import os
# dmf.import_lib()
import matplotlib.pyplot as plt
import scipy
import statistics
import scipy.signal
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eig, eigh



class PossibleCluster:
    def __init__(self):
        self.data = []
        self.labels = []
        self.eigen_vectors = []
        self.length = 0
        self.min_entry_dist = 0


class DiffuseMap:
    def __init__(self, encodings_file, eps, load=False, pickle_file=''):
        self.b = 0
        self.eps = eps
        self.unknown_clusters = 0
        self.clusters = {}
        self.possible_clusters = {}
        self.cluster_names = []
        self.clusters_entry_dists = {}
        self.eigen_vectors = None
        self.eigen_values = None
        self.routine_dist=0

        if not load:
            self.raw_data = pickle.loads(open(encodings_file, "rb").read())
            self.data = np.array(self.raw_data['encodings'])
            self.labels = np.array(self.raw_data['names'])
            self.update_diff_map()
            self.save_pickle(pickle_file)




        else:
            self.load_from_pickle(pickle_file)

    def DM_distance(self,u,v):
        if self.routine_dist==0:
            d=np.exp(-(np.linalg.norm(u - v) ** 2 / self.eps))
        elif self.routine_dist==1:
            #d = cos...
            pass
        return d

    def update_diff_map(self, new_points=None, new_labels=None):

        if new_points is not None:
            new_points = np.array(new_points)
            self.data = np.concatenate((self.data, new_points))
        if new_labels is not None:
            new_labels = np.array(new_labels)
            self.labels = np.concatenate((self.labels, new_labels))

        self.calc_eigen_v(self.get_p())
        (r, c) = self.eigen_vectors.shape
        labels = set(self.labels)
        for l in labels:
            ids = np.where(self.labels == l)
            cluster_ = [self.eigen_vectors[:, i] for i in ids][0]
            self.clusters[l] = cluster_
            self.clusters_entry_dists[l] = 2 * self.calc_cluster_entry_dist(cluster_)

    def get_p(self):
        p_matrix = squareform(pdist(self.data, lambda u, v: self.DM_distance(u,v)))  # Гауссовское ядро
        return p_matrix

    def calc_eigen_v(self, Q):
        lambda_, psi_ = np.linalg.eig(Q)
        lambda_ = list(lambda_)
        map_projection = lambda_.copy()
        map_projection.sort(reverse=True)
        pos = [lambda_.index(j) for j in map_projection[0:3]]
        eigen_vectors = np.array([map_projection[0] * psi_[:, pos[0]], map_projection[1] * psi_[:, pos[1]], map_projection[2] * psi_[:, pos[2]]])

        self.eigen_vectors = eigen_vectors
        self.eigen_values = map_projection
        (r, c) = self.eigen_vectors.shape

        return eigen_vectors, map_projection

    def insert_base(self, new):
        # print('new: ', len(new),new[-1].shape)
        # print('self data: ', len(self.data))
        data1 = []
        for u in new:
            data2 = []
            for v in self.data:
                try:
                    data2.append(self.DM_distance(u, v))
                except:
                    pass
            data1.append(np.array(data2))
        data1 = np.array(data1)

        results = []
        # print(data1.shape)
        for d in data1:
            try:
                results.append([np.dot(d, self.eigen_vectors[0, :].T) / self.eigen_values[0],
                                np.dot(d, self.eigen_vectors[1, :].T) / self.eigen_values[1],
                                np.dot(d, self.eigen_vectors[2, :].T) / self.eigen_values[2]])
            except ValueError:
                print()
        results = np.array(results)

        return results


    def insert(self, new, visualize=False):
        data1 = []
        for u in new:
            data2 = []
            for v in self.data:
                try:
                    data2.append(self.DM_distance(u,v))
                except:
                    pass
            data1.append(np.array(data2))
        data1 = np.array(data1)

        results = []
        for d in data1:
            results.append([np.dot(d, self.eigen_vectors[0, :].T) / self.eigen_values[0],
                            np.dot(d, self.eigen_vectors[1, :].T) / self.eigen_values[1],
                            np.dot(d, self.eigen_vectors[2, :].T) / self.eigen_values[2]])
        results = np.array(results)

        # add to cluster

        labels_of_given_vectors = []
        for i, dot in enumerate(results):
            # if i > 3:
            #     break
            v = dot.reshape(len(dot), 1)

            # (k, vls) = self.clusters.popitem()
            # m1 = self.mahalanobis(v, vls)
            # m2 = self.mahalanobis(v, vls)

            dists = {l: self.mahalanobis(v, c) for (l, c) in self.clusters.items()} # [0]

            dists_key_list = list(dists.keys())
            dists_value_list = list(dists.values())

            min_dist = min(dists_value_list)
            nearest_cluster = dists_key_list[dists_value_list.index(min_dist)]
            cluster_median_dist = self.clusters_entry_dists[nearest_cluster]
            clusters_mean_dist = statistics.mean(self.clusters_entry_dists.values())
            if min_dist <= cluster_median_dist:
                if self.possible_clusters.get(nearest_cluster) is not None:
                    tc = self.possible_clusters[nearest_cluster]
                    tc.data.append(new[i])
                    tc.eigen_vectors = np.append(tc.eigen_vectors, v, axis=1)
                    tc.labels.append(nearest_cluster)
                    tc.length += 1
                    if tc.length >= 4:
                        new_min_dist = self.calc_cluster_entry_dist(tc.eigen_vectors)
                        tc.min_entry_dist = new_min_dist

                    self.possible_clusters[nearest_cluster] = tc
                    labels_of_given_vectors.append(nearest_cluster)

                else:
                    new_cluster = PossibleCluster()
                    new_cluster.eigen_vectors = v
                    new_cluster.data.append(new[i])
                    new_cluster.labels.append(nearest_cluster)
                    new_cluster.length = 1
                    new_cluster.min_entry_dist = clusters_mean_dist
                    self.possible_clusters[nearest_cluster] = new_cluster
                    labels_of_given_vectors.append(nearest_cluster)
                # self.plot3d(50, dot, i, min_dist, nearest_cluster, '../test13/')
            else:
                if len(self.possible_clusters.values()) != 0:
                    p_dists = {}
                    for (l, c) in self.possible_clusters.items():
                        if c.eigen_vectors.shape[1] < 15:
                            p_dists[l] = np.linalg.norm(c.eigen_vectors[:, -1] - v[:, 0])
                        else:
                            p_dists[l] = self.mahalanobis(v, c.eigen_vectors)  #[0]

                    p_dists_key_list = list(p_dists.keys())
                    p_dists_value_list = list(p_dists.values())

                    p_min_dist = min(p_dists_value_list)
                    p_nearest_cluster = p_dists_key_list[p_dists_value_list.index(p_min_dist)]
                    p_clusters_mean_dist = \
                        statistics.mean(list(x.min_entry_dist for x in self.possible_clusters.values()))
                    if p_min_dist <= 5 * self.possible_clusters[p_nearest_cluster].min_entry_dist:
                        tc = self.possible_clusters[p_nearest_cluster]
                        tc.data.append(new[i])
                        tc.labels.append(p_nearest_cluster)
                        tc.eigen_vectors = np.append(tc.eigen_vectors, v, axis=1)
                        tc.length += 1
                        if tc.length >= 4:
                            new_min_dist = self.calc_cluster_entry_dist(tc.eigen_vectors)
                            tc.min_entry_dist = new_min_dist
                        self.possible_clusters[p_nearest_cluster] = tc
                        labels_of_given_vectors.append(p_nearest_cluster)
                    else:
                        new_label = 'New_class' + str(self.unknown_clusters)
                        new_cluster = PossibleCluster()
                        new_cluster.eigen_vectors = v
                        new_cluster.data.append(new[i])
                        new_cluster.labels.append(new_label)
                        new_cluster.length = 1
                        new_cluster.min_entry_dist = clusters_mean_dist
                        self.possible_clusters[new_label] = new_cluster
                        self.unknown_clusters += 1
                        labels_of_given_vectors.append(new_label)
                else:
                    new_label = 'New_class' + str(self.unknown_clusters)
                    new_cluster = PossibleCluster()
                    new_cluster.eigen_vectors = v
                    new_cluster.data.append(new[i])
                    new_cluster.labels.append(new_label)
                    new_cluster.length = 1
                    new_cluster.min_entry_dist = clusters_mean_dist
                    self.possible_clusters[new_label] = new_cluster
                    self.unknown_clusters += 1
                    labels_of_given_vectors.append(new_label)
            if visualize:
                self.plot_diff_map()


        pc_to_del = []
        for (l, c) in self.possible_clusters.items():
            if c.length >= 50:
                self.update_diff_map(c.data, c.labels)
                pc_to_del.append(l)
                if visualize:
                    self.plot_diff_map()

        for cl in pc_to_del:
            del self.possible_clusters[cl]

        return labels_of_given_vectors

    def vizu_with_labels(self, new, new_labels, index=None, camera_image=None, res_path_out="./plot_out/"):

        embedded_ = self.insert_base(new)  # new: kx128  embedded_:kx3 new_labels:kx1
        res_dict = {}
        l_set = set(new_labels)
        for l in l_set:
            res_dict[l] = []
        for (l, v) in zip(new_labels, embedded_):
            res_dict[l].append(v)

        self.plot3d_0(res_dict, None, camera_image, res_path_out)

    def quasy_inv_matr(self, collect_feature, qqq=10**(-10)):
        U, S, V = np.linalg.svd(collect_feature)
        dd = np.diag(np.diag(S))
        S = np.diag(S)
        dd2 = np.multiply(np.sign(dd), np.maximum(qqq, np.abs(dd)))
        dd_ = np.divide(1, dd2 + np.finfo(float).eps)
        S_ = np.diag(dd_)
        S1 = np.transpose(S)
        S1[0:S_.shape[1]][0:S_.shape[0]] = S_
        return np.matmul(np.matmul(V.T, S1), U.T)

    def mahalanobis(self, y, x=None):
        if x is None:
            x = self.eigen_vectors
        x = x.T

        y = y.T
        (rx, cx) = x.shape
        (ry, cy) = y.shape

        if cx != cy:
            raise Exception('stats:mahal:InputSizeMismatch')
        if rx < cx:
            raise Exception('stats:mahal:TooFewRows')
        X = np.vstack([x, y])
        V = np.cov(X.T)
        # VI = np.linalg.inv(V)
        VI = self.quasy_inv_matr(V, qqq=1)
        A = np.dot((x - y), VI)
        B = (x - y).T
        D = np.sqrt(np.einsum('ij,ji->i', A, B))
        return np.abs(np.mean(D))

    def calc_cluster_entry_dist(self, cluster):
        dists = []
        temp_cluster = cluster.T
        for i, elem in enumerate(temp_cluster):
            cclust = temp_cluster[np.arange(len(temp_cluster)) != i]
            # elem = np.array(elem)
            elem = elem.reshape(len(elem), 1)
            res = self.mahalanobis(elem, cclust.T)
            dists.append(res) # [0]
        return statistics.median(dists)

    def test_for_anomaly(self, cluster, target_vector):
        """ True - no anomaly, False - anomaly"""

        embedded_cluster = self.insert_base(cluster)
        min_dist = self.calc_cluster_entry_dist(embedded_cluster.T)

        embedded_vector = self.insert_base(target_vector)
        dist = self.mahalanobis(embedded_vector.T, embedded_cluster.T)
        print('Test for anomaly: ', dist < min_dist)
        return dist < min_dist


    def plot3d(self, k, new_dots, index, mah_dist, near_cluster, images_dir, res_path_out="./plot_out/"):
        fig = plt.figure(figsize=(24, 10))
        ax2 = fig.add_subplot(1, 2, 1)
        image = mpimg.imread(images_dir + str(index).zfill(5) +'.jpg')

        ax = fig.add_subplot(1, 2, 2, projection='3d')

        for lbl, d in self.clusters.items():
            d = d.T
            ax.scatter(d[:, 0], d[:, 1], d[:, 2], label=lbl)

        if near_cluster is not None:
            ax.set_title('Diffuse map, \nMin Mahalanobis dist = ' + str(mah_dist) + '; \nNear cluster: ' + near_cluster,
                         pad=30)
        else:
            ax.set_title('Diffuse map, \nMin Mahalanobis dist = ' + str(mah_dist) + '; \nNear cluster: None',
                         pad=30)


        if new_dots.ndim == 1:
            ax.scatter(new_dots[0], new_dots[1], new_dots[2], label='new face')
        else:
            ax.scatter(new_dots[:, 0], new_dots[:, 1], new_dots[:, 2], label='new face')

        ax2.imshow(image)
        plt.legend()
        plt.show()
        plt.savefig(res_path_out + str(index).zfill(5) + '.jpg')


    def plot3d_0(self, dict_with_vectors, index=None, camera_image=None, res_path_out="./plot_out/"):
        if not os.path.exists(res_path_out):
            os.makedirs(res_path_out)

        if camera_image is not None:
            fig = plt.figure(figsize=(24, 10))
            ax2 = fig.add_subplot(1, 2, 1)
            image = camera_image

            ax = fig.add_subplot(1, 2, 2, projection='3d')

            for lbl, d in dict_with_vectors.items():
                # d = d.T
                d = np.array(d)
                ax.scatter(d[:, 0], d[:, 1], d[:, 2], label=lbl)
            ax.set_title('Diffuse map', pad=30)

            ax2.imshow(image)
            plt.legend()
            # plt.show()
            if index is None:
                index = self.b
            plt.savefig(res_path_out + str(index).zfill(5) + '.jpg')

            self.b += 1
            plt.close(fig)
        else:
            pass

    def plot_diff_map(self, index=None, camera_image=None, res_path_out="./plot_out/"):

        if not os.path.exists(res_path_out):
            os.makedirs(res_path_out)

        if camera_image is not None:
            fig = plt.figure(figsize=(24, 10))
            ax2 = fig.add_subplot(1, 2, 1)
            image = camera_image

            ax = fig.add_subplot(1, 2, 2, projection='3d')

            # for lbl, d in self.clusters.items():
            #     d = d.T
            #     ax.scatter(d[:, 0], d[:, 1], d[:, 2], label=lbl)

            for lbl, d in self.possible_clusters.items():
                d = d.eigen_vectors.T
                ax.scatter(d[:, 0], d[:, 1], d[:, 2], label=lbl)
            ax.set_title('Diffuse map', pad=30)

            ax2.imshow(image)
            plt.legend()
            # plt.show()
            if index is None:
                index = self.b
            plt.savefig(res_path_out + str(index).zfill(5) + '.jpg')

            self.b += 1
            plt.close(fig)
        else:
            fig = plt.figure(figsize=(16, 10))
            ax = fig.add_subplot(1, 1, 1, projection='3d')

            for lbl, d in self.clusters.items():
                d = d.T
                ax.scatter(d[:, 0], d[:, 1], d[:, 2], label=lbl)

            for lbl, d in self.possible_clusters.items():
                d = d.eigen_vectors.T
                ax.scatter(d[:, 0], d[:, 1], d[:, 2], label=lbl)
            ax.set_title('Diffuse map', pad=30)

            plt.legend()
            # plt.show()
            if index is None:
                index = self.b
            plt.savefig(res_path_out + str(index).zfill(5) + '.jpg')

            self.b += 1
            plt.close(fig)

    def save_pickle(self, file_name):
        with open(file_name, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_from_pickle(self, file_name):
        import os
        print(os.getcwd())
        with open(file_name, 'rb') as handle:
            DM = pickle.load(handle)
            self.data = DM.data
            self.possible_clusters = DM.possible_clusters
            self.labels = DM.labels
            self.clusters = DM.clusters
            self.eigen_vectors = DM.eigen_vectors
            self.eigen_values = DM.eigen_values
            self.clusters_entry_dists = DM.clusters_entry_dists
            self.unknown_clusters = DM.unknown_clusters
            self.cluster_names = DM.cluster_names
            self.eps = DM.eps

    def calcMetric(self):
        cluster_mean = []
        cluster_var = []
        print("Clusters: ", len(self.clusters.items()))
        print("Possible_clusters: ", len(self.possible_clusters.items()))

        for lbl, d in self.clusters.items():
            m = np.mean(d.T, axis=0)
            cluster_mean.append(m)
            cluster_var.append(np.mean(np.square(d.T - m)))

        c_m = np.mean(cluster_mean, axis=0)
        return np.mean(np.square(cluster_mean - c_m)) / np.mean(cluster_var)


def showDiffMap(model, test_encodings):
    print('Reading DATASET', test_encodings)
    test_face_data = pickle.loads(open(test_encodings, "rb").read())
    count_data = test_face_data['names']
    print(f'TEST DATASET include {len(count_data)} items.')
    test_data = np.array(test_face_data['encodings'])

    diff_map = DiffuseMap(test_encodings, 0.8, show=0)

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    for lbl, d in diff_map.clusters.items():
        d = d.T
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], label=lbl)

    for lbl, d in diff_map.possible_clusters.items():
        d = d.eigen_vectors.T
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], label=lbl)

    ax.set_title('Diffuse map {}'.format(model), pad=30)
    ax.view_init(azim=30)
    plt.legend()
    plt.show()
    return diff_map
