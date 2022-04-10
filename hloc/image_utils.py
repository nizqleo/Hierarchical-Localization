import numpy as np

from descartes import PolygonPatch
import alphashape

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import os, logging, h5py, pickle, cv2

from PIL import Image, ImageDraw
from tqdm import tqdm
from collections import defaultdict
from scipy import spatial
import pycolmap
import time

import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.transform import Rotation as R
import seaborn as sns

import logging
from datetime import datetime

from scipy.spatial.distance import cdist
import json


def similarity_measurement(image1, image2, inters, unions):
    return len(inters)/len(unions)

def image_similarity_computation(images, points3D, path):
    
    index2image_id = dict()
    image_id2index = dict()
    cnt = 0
    for image_id in tqdm(images):
        index2image_id[cnt] = image_id
        image_id2index[image_id] = cnt
        cnt += 1
        
    if os.path.exists(path):
        affinity_matrix = np.load(path)
        return affinity_matrix , index2image_id
    
    image_num = len(images)
   
    affinity_matrix = np.zeros((image_num, image_num))
    
    visited = set()
    for image_id in tqdm(images):
        observed = images[image_id].point3D_ids
        observed = set(observed)
        co_vis_images = set(j for i in observed if i != -1 for j in points3D[i].image_ids)
        for co_vis_id in co_vis_images:
            if co_vis_id in visited:
                continue
            inters = set(images[co_vis_id].point3D_ids).intersection(observed)
            unions = set(images[co_vis_id].point3D_ids).union(observed)
            s = similarity_measurement(images[image_id], images[co_vis_id], inters, unions)
            affinity_matrix[image_id2index[co_vis_id], image_id2index[image_id]] = affinity_matrix[image_id2index[image_id], image_id2index[co_vis_id]] = s
        visited.add(image_id)
        
    np.save(path, affinity_matrix)
    return affinity_matrix, index2image_id


def image_co_vis_grouping(images, points3D, group_result_dir, n_cluster = 2000):

    labels_path = group_result_dir / "cluster_result.pkl"
    id_path = group_result_dir / "index2image_id.pkl"
    
    affinity_matrix, index2image_id = image_similarity_computation(images, points3D, "group/aachen/a_matrix.npy")
    distance_matrix = 1 - affinity_matrix
    aggclus = AgglomerativeClustering(n_clusters=n_cluster, 
                                      memory= str(str(group_result_dir) + "/temp"), 
                                      linkage="complete",
                                      affinity = "precomputed").fit(distance_matrix)
    labels = aggclus.labels_
     
    with open(labels_path, "wb") as f:
        pickle.dump(labels, f)
    with open(id_path, "wb") as f:
        pickle.dump(index2image_id, f)
    return labels, index2image_id


def image_desc_grouping(global_feature_path, images, group_result_dir, n_cluster = 2000):
    db_desc = h5py.File(global_feature_path, 'r')

    labels_path = group_result_dir / "cluster_result.pkl"
    id_path = group_result_dir / "index2image_id.pkl"
    
    group_desc = []
    index2image_id = dict()
    ind = 0
    for image_id, image in tqdm(images.items()):
        group_desc.append(np.array(db_desc[image.name]['global_descriptor']))
        index2image_id[ind] = image_id
        ind += 1

    print(len(group_desc))
    group_desc=np.array(group_desc)
    
    aggclus = AgglomerativeClustering(n_clusters=n_cluster, memory=str(str(group_result_dir) + "/temp")).fit(group_desc)
    # labels = kmeans.predict(group_desc)
    labels = aggclus.labels_
    
    with open(labels_path, "wb") as f:
        pickle.dump(labels, f)
    with open(id_path, "wb") as f:
        pickle.dump(index2image_id, f)
        
    return labels, index2image_id

def points_grouping(images, labels, index2image_id, image_global_feature_path):
    groups_image_points = defaultdict(set) 
    group_descriptor = defaultdict(list) 
    
    db_desc = h5py.File(image_global_feature_path, 'r')
    
    for i in tqdm(range(len(labels))):
        image = images[index2image_id[i]]
        p3d_id_set = set(image.point3D_ids)        
        groups_image_points[labels[i]] = groups_image_points[labels[i]].union(p3d_id_set) 
        group_descriptor[labels[i]].append(np.array(db_desc[image.name]['global_descriptor']))
        
    for label, desc in tqdm(group_descriptor.items()):
        if len(desc) == 0:
            group_descriptor.pop(label, None)
            continue
        arr = np.array(desc)
        
        arr = np.median(arr, axis=0)
        group_descriptor[label] = arr
#         dist = cdist(arr, np.mean(arr, axis=0).reshape((1,4096)), metric='euclidean')
#         dist_idx = np.argmin(dist, axis=0)
#         group_descriptor[label] = arr[dist_idx].reshape(4096)
        
    cluster_point_num = []
    for gid, cps in groups_image_points.items():
        if len(cps) == 1:
            print(gid)
        cluster_point_num.append(len(cps))
        
    somelist_df = pd.DataFrame(cluster_point_num)
    print(somelist_df.describe())
        
    return groups_image_points, group_descriptor
        
def points_grouping_test(images, labels, index2image_id, image_global_feature_path):
    group_descriptor = defaultdict(list) 
    db_desc = h5py.File(image_global_feature_path, 'r')
    
    for i in tqdm(range(len(labels))):
        image = images[index2image_id[i]]    
        group_descriptor[labels[i]].append(np.array(db_desc[image.name]['global_descriptor']))
        
    for label, desc in tqdm(group_descriptor.items()):
        if len(desc) == 0:
            group_descriptor.pop(label, None)
            continue
        arr = np.array(desc)
        group_descriptor[label] = np.mean(arr, axis=0)
#         dist = cdist(arr, np.median(arr, axis=0).reshape((1,4096)), metric='cosine')
#         dist_idx = np.argmin(dist, axis=0)
#         group_descriptor[label] = arr[dist_idx].reshape(4096)
    return  group_descriptor

def build_tree(group_descriptor, nca = None):
    group_desc = []
    index2name = dict()
    ind = 0
    for gid, desc in tqdm(group_descriptor.items()):
        group_desc.append(desc)
        index2name[ind] = gid
        ind += 1

    group_desc=np.array(group_desc)
    if nca is not None:
        group_desc = nca.transform(group_desc)

    tree = spatial.KDTree(group_desc)
    
    return tree, index2name


def generate_retrieval_file(global_feature_path, tree, index2name, query_images, retrieval_dir, 
                            nca = None, label = "whole", top_n = 50):
    desc = h5py.File(global_feature_path, 'r')

    retrieval_path = retrieval_dir / str("retrieval_" + label + "_"+str(top_n)+".txt")
    file = open(retrieval_path, "w+")

    for query_image in tqdm(query_images):
        name, param = query_image
        cur_desc = np.array([desc[name]['global_descriptor']])[0]
        if nca is not None:
            cur_desc = nca.transform(cur_desc)[0]
        d, group_indices = tree.query(cur_desc, top_n)
        if top_n == 1:
            file.write(name + ' ' + str(index2name[group_indices]) + '\n') 
        else:
            for i in range(len(group_indices)):
                file.write(name + ' ' + str(index2name[group_indices[i]]) + '\n')            

def visualization(images, labels, show_labels, size = (6,6)):
    image_x = []
    image_y = []
    other_label = None
    if show_labels is not None:
        for i in range(len(labels)):
            if labels[i] not in show_labels:
                if other_label is None:
                    other_label = labels[i]
                labels[i] = other_label
        
    class_num = len(set(labels))
    
    cnt = 0
    bg_x = []
    bg_y = []
    hl_labels = []
    for image_id, image in images.items():
        qvec = image.qvec
        tvec = np.array(image.tvec)
        r = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])
        t = -r.inv().apply(tvec)
        
        if labels[cnt] not in show_labels:
            bg_x.append(t[0])
            bg_y.append(t[2])
        else:
            image_x.append(t[0])
            image_y.append(t[2])
            hl_labels.append(labels[cnt])
            
        cnt +=1 
        
    print(len(image_x), len(image_y), len(labels))
    plt.figure(figsize=size)
    sns.scatterplot(
        x=image_x, y=image_y,
        hue=hl_labels,
        palette=sns.color_palette("hls", len(show_labels)),
        legend="full",
        alpha=1,
        s=70
    )
    
    
    sns.scatterplot(
        x=bg_x, y=bg_y,
        color=".01", 
        legend="full",
        alpha=0.05,
        s=5
    )
    
    plt.savefig("visual.pdf", format='pdf')
    
#     if show_labels is not None:

#         for image_id in centers:
#             image_id += 1
#             qvec = images[image_id].qvec
#             tvec = np.array(images[image_id].tvec)
#             r = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])
#             t = -r.inv().apply(tvec)

#             bg_x.append(t[0])
#             bg_y.append(t[1])


#     if show_labels is not None:
#         sns.scatterplot(
#             x=bg_x, y=bg_y,
#             alpha=0.1,
#             legend=False,
#             s=150
#         )
        
        
def localize(query_images, test_retrieve_groups_file, local_feature_path, point_desc_path, 
             groups_image_points, points3D_new, images_new, cameras_new, log_path):
    
    print("localizing:", log_path)
    
    # read retrieval result
    point_descs = dict()
    with open(point_desc_path, "rb") as file:
        point_descs = pickle.load(file)
#     logging.info("loading point descs2")
    
#     with open(str(point_desc_path)[:-4] + "_2.pkl", "rb") as file:
#         temp = pickle.load(file)
#     logging.info("merging")
    
#     point_descs.update(temp)
#     del temp
    
    retrieval_file = open(test_retrieve_groups_file, "r")
    test_retrieve_groups = defaultdict(list) 
    for line in retrieval_file:
        line = line.split()
        test_retrieve_groups[line[0]].append(int(line[1]))
        
    
    f = h5py.File(local_feature_path, 'r')

#     FLANN_INDEX_LINEAR 			= 0,
#     FLANN_INDEX_KDTREE 			= 1,
#     FLANN_INDEX_KMEANS 			= 2,
#     FLANN_INDEX_COMPOSITE 		= 3,
#     FLANN_INDEX_KDTREE_SINGLE 	= 4,
#     FLANN_INDEX_HIERARCHICAL 	= 5,
#     FLANN_INDEX_LSH 			= 6,
    
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    matcher = cv2.FlannBasedMatcher(index_params,search_params)
    
#     matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    

#     point_desc_file = h5py.File(point_desc_path, 'a')

    result = dict()
    max_inlier_nums = dict()
    
    cluster_time = 0
    load_time = 0
    new_match_time = 0
    match_time = 0
    ratio_time = 0
    pose_time = 0
    
    total_s = time.time()
    
    infos = dict()
    for query_image in tqdm(query_images):
        image_name, param = query_image
        infos
        if image_name in result:
            continue
        candidate_point_set = list()
        groups = test_retrieve_groups[image_name]
        
        s = time.time()
        for group in groups:
            merged_flag = False
            
            for i in range(len(candidate_point_set)):
                if len(candidate_point_set[i].intersection(groups_image_points[group])) > 100:
                    merged_flag = True
                    candidate_point_set[i] = candidate_point_set[i].union(groups_image_points[group])
                    break
            if not merged_flag:
                candidate_point_set.append(groups_image_points[group])

        cluster_time += time.time() - s
        
        max_inlier_num = 0
        final_qvec = np.array([-0.02718695,-0.12122045,0.20194369,0.97148609])
        final_tvec = np.array([723.98874206,76.41164933,187.85558868])
        
        candidate_point_set = sorted(candidate_point_set, key=len, reverse=True)
        
        cluster_infos = list()
        # fusion & matching
        for cps in candidate_point_set:
            cluster_info = dict()
            s = time.time()
            p3d_descs = list()
            p3d_ids = list()
            for p3d_id in cps:
                if p3d_id == -1:
                    continue
                p3d_ids.append(p3d_id)
                sid = str(p3d_id)
                p3d_descs.append(point_descs[p3d_id])
            
            if len(p3d_descs) < 8:
                continue
            
            p3d_descs = np.array(p3d_descs)
            p2d_descs = f[image_name]['descriptors'].__array__().transpose()
            
            load_time += time.time() - s
            s = time.time()
            matches = matcher.knnMatch(p2d_descs, p3d_descs, k=2)
            match_time += time.time() - s
            
            s = time.time()
            mp3d = list()
            mkpq = list()
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.8*n.distance:
                    mp3d.append(points3D_new[p3d_ids[m.trainIdx]].xyz)
                    mkpq.append(f[image_name]['keypoints'][m.queryIdx])

            mp3d = np.array(mp3d).reshape(-1, 3)
            mkpq = np.array(mkpq).reshape(-1, 2)
            cluster_info["#matches"] = mp3d.shape[0]
            mkpq += 0.5

            ratio_time += time.time() - s
            s = time.time()
            cfg = {
                'model': param[0],
                'width': param[1],
                'height': param[2],
                'params': param[3],
            }
            ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg)
            cluster_info["suc"] = ret['success']
            
            if ret['success'] == True:
                cluster_info["#inliers"] = ret['num_inliers']
                if ret['num_inliers'] > max_inlier_num:
                    max_inlier_num = ret['num_inliers']
                    final_qvec = ret['qvec']
                    final_tvec = ret['tvec']
            pose_time += time.time() - s
            cluster_infos.append(cluster_info)
            
        result[image_name] = (final_qvec, final_tvec)
        max_inlier_nums[image_name] = max_inlier_num
        
        infos[image_name] = {"max_inlier_nums":max_inlier_num, 
                             "#candidate" : len(candidate_point_set),
                             "candidates" : cluster_infos,
                             "max_inlier_nums": max_inlier_num
                            } 
    print(cluster_time, match_time, new_match_time, load_time, ratio_time, pose_time, time.time()-total_s)
    
    with open(log_path, 'w') as fp:
        json.dump(infos, fp)
    f.close()
    del point_descs
#     point_desc_file.close()

    return result, max_inlier_nums