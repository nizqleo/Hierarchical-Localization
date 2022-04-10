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

def box_preprocess(data, w, h):
#     print(w, h, len(data))
    data = list(data)
    data.sort(key = lambda x: x[1])
    remove_list = list()
    
    for i in range( len(data) - 1):
#         print(i, data[i])
        if data[i][0] == -1:
            continue
        if data[i+1][1] - data[i][1] < 0.1: # close left edge
            if abs(data[i+1][3] - data[i][3]) > 0.2:# large right edge dist: split
                if data[i+1][3] > data[i][3]:
                    data[i+1][1] = data[i][3]
                else:
                    data[i][1] = data[i+1][3]
                    
            else:# merge
                if data[i][2] - data[i][0] > data[i+1][2] - data[i+1][0]:
                    data[i][0] = -1
                else:
                    data[i+1][0] = -1
        if abs(data[i+1][3] - data[i][3]) < 0.1: # close right edge
            if abs(data[i+1][1] - data[i][1]) > 0.2:# large left edge dist: split
                if data[i+1][1] > data[i][1]:
                    data[i][3] = data[i+1][1]
                else:
                    data[i+1][3] = data[i][1]
                    
            else:# merge
                if data[i][2] - data[i][0] > data[i+1][2] - data[i+1][0]:
                    data[i][0] = -1
                else:
                    data[i+1][0] = -1
                    
#         if (data[i][2] - data[i][0])*h < 0.3*(data[i][3] - data[i][1])*w:
#             data[i][0] = -1
            
#     if len(data) > 0:
#         if (data[-1][2] - data[-1][0])*h < 0.3*(data[-1][3] - data[-1][1])*w:
#             data[-1][0] = -1

    data_new = [x for x in data if x[0] > -1]
    return np.array(data_new)




class PatchSet:
    def __init__(self):
        self.fa = list()
        pass

    def find(self, x):
        if x == self.fa[x]:
            return x
        else:
            self.fa[x] = self.find(self.fa[x])
            return self.fa[x];

    def merge(self, i, j):
        self.fa[self.find(j)] = self.find(i)
        
        
        
        
        
class Patch:
    alphashape_thres = 0.5

    def __init__(self, box, image_id, ind):
        self.box = box
        self.image_id = image_id
        self.p3d_ids = set()
        self.p2d_xys = dict()
        self.validaty = False
        self.area = (self.box[2] - self.box[0])*(self.box[3] - self.box[1])
        pass

    def check_validation(self, xys, p3d_ids):
        whole_xys = list()
        for i in range(len(xys)):
            if (xys[i][1] > self.box[0] and xys[i][1] < self.box[2] and 
                xys[i][0] > self.box[1] and xys[i][0] < self.box[3]):
                whole_xys.append(xys[i])
                self.p3d_ids.add(p3d_ids[i])
                self.p2d_xys[p3d_ids[i]] = xys[i]
            
        AS = alphashape.alphashape(whole_xys,10)  
        
        rate = AS.area/self.area
        if rate >= self.alphashape_thres:
            self.validaty = True
        else:
            self.validaty = False
            
        return self.validaty

    def check_match(self, patch_B, show = False):
        inters = self.p3d_ids.intersection(patch_B.p3d_ids)
        inters = list(inters)
        if show:
            print(inters)
        if len(inters) < 25:
            return False
        if show:
            print (self.check_coverage(inters, True), patch_B.check_coverage(inters))
        if (self.check_coverage(inters) > self.alphashape_thres*0.6 and 
           patch_B.check_coverage(inters) > self.alphashape_thres*0.6):
            return True
        else:
            return False

    def save_crop(self, image_path, save_path):
        im = Image.open(image_path)
        width, height = im.size
        im1 = im.crop((self.box[1]*width, self.box[0]*height, self.box[3]*width, self.box[2]*height))
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        im1.save(save_path)

    def check_coverage(self, p3d_ids, show = False):
        xys = [self.p2d_xys.get(key) for key in p3d_ids]
        AS = alphashape.alphashape(xys,5)  
        if show:
            print(AS.area, self.area, self.image_id)
            fig, ax = plt.subplots()
            ax.scatter(*zip(*xys))
            # Plot alpha shape
            ax.add_patch(PolygonPatch(AS, alpha=.2))
            ax.set_ylim(self.box[1],self.box[3])
            ax.set_xlim(self.box[0],self.box[2])
            
            plt.savefig("alphashape.png")
            
        return AS.area/self.area
    def compute_similarity(self, patch_a):
        inters_len = len(self.p3d_ids.intersection(patch_a.p3d_ids))
        unions_len = len(self.p3d_ids.union(patch_a.p3d_ids))
        return inters_len/unions_len
    
    
    
def test_generate_patches(images, box_dir, test_image2patch_path, image_dir, do_croping = False, crop_output_dir = None):
    patch_list = list()
    image2patch = dict()
    
    # for error reporting
    no_box_detected_images = list()
    no_box_left_images = list()
    
    patch_cnt = 0
    
    for image in tqdm(images):
        file_name, param = image
        image_name = file_name.split('/')[-1]
        box_file_name = file_name.split('.')[0] + ".npy"

        image2patch[image_name] = list()
        
        w = param[1]
        h = param[2]
        
        if not os.path.exists(os.path.join(box_dir, box_file_name)):
            logging.warning("no bbox found for "+ os.path.join(box_dir, box_file_name))
            continue
            
        data = np.load(os.path.join(box_dir, box_file_name))
        if data.shape[0] == 0:
            no_box_detected_images.append(file_name)
            logging.warning("no object detected for "+ os.path.join(box_dir, box_file_name))
            continue
            
        data=box_preprocess(data, w, h)
        if data.shape[0] == 0:
            no_box_left_images.append(file_name)
            continue
            
        for box_id in range(data.shape[0]):
            temp_patch = Patch(data[box_id], image_name, patch_cnt)
            patch_list.append(temp_patch)
            image2patch[image_name].append(patch_cnt)
            
            if do_croping:    
                image_path = os.path.join(image_dir, file_name)
                save_path = os.path.join(crop_output_dir, str(patch_cnt)+ ".png")
                temp_patch.save_crop(image_path, save_path)
                
            patch_cnt += 1

    print("number of patches:", len(patch_list))  
    
    with open(test_image2patch_path, "wb") as f:
        pickle.dump(image2patch,f)
        
    for i in range(len(patch_list)):
        del patch_list[i].p2d_xys
       
    with open("./aachen_test_patch_list.pkl", "wb") as f:
        pickle.dump(patch_list,f)
        
    return patch_list, image2patch


def database_generate_patches(images, cameras, box_dir, image2patch_path):
    patch_list = list()
    image2patch = dict()
    
    # for error reporting
    no_box_detected_images = list()
    no_box_left_images = list()
    
    patch_cnt = 0
    
    for image_id, image in tqdm(images.items()):

        box_file_name = image.name.split('.')[0] + ".npy"

        image2patch[image.name] = list()

        matched = image.point3D_ids != -1
        points3D_covis = image.point3D_ids[matched]
        total_matched_p2d_xys = image.xys[matched]
        
        w = cameras[image.camera_id].width
        h = cameras[image.camera_id].height

        # p2d xy normalization 
        for i in range(len(total_matched_p2d_xys)):
            total_matched_p2d_xys[i][0] /= w
            total_matched_p2d_xys[i][1] /= h

        if not os.path.exists(os.path.join(box_dir, box_file_name)):
#             logging.warning("no bbox found for "+ os.path.join(box_dir, box_file_name))
            continue
            
        data = np.load(os.path.join(box_dir, box_file_name))
        if data.shape[0] == 0:
            no_box_detected_images.append(image.name)
            logging.warning("no object detected for "+ os.path.join(box_dir, box_file_name))
            continue
            
        data=box_preprocess(data, w, h)
        if data.shape[0] == 0:
            no_box_left_images.append(image.name)
            continue
            
        for box_id in range(data.shape[0]):
            temp_patch = Patch(data[box_id], image_id, patch_cnt)
            validity = temp_patch.check_validation(total_matched_p2d_xys, points3D_covis)#, os.path.join(image_dir, image.name))
#             if validity:
            patch_list.append(temp_patch)
            image2patch[image.name].append(patch_cnt)
            patch_cnt += 1

    print("number of patches:", len(patch_list))  
    
    with open(image2patch_path, "wb") as f:
        pickle.dump(image2patch,f)
       
    for i in range(len(patch_list)):
        del patch_list[i].p2d_xys
       
    with open("./aachen_patch_list.pkl", "wb") as f:
        pickle.dump(patch_list,f)
    
    return patch_list


########################

def patch_similarity_computation(patch_list, db_image_pairs_path, image2patch_path, save_path):
    if os.path.exists(save_path):
        affinity_matrix = np.load(save_path)
        return affinity_matrix
    
    with open(db_image_pairs_path, "r") as f:
        db_image_pairs = f.readlines()
    with open(image2patch_path, "rb") as f:
        image2patch = pickle.load(f)

    patch_num = len(patch_list)
    affinity_matrix = np.zeros((patch_num, patch_num))
    
    for pair in tqdm(db_image_pairs):
        image1 = pair.split()[0]
        image2 = pair.split()[1]
        if image1 < image2:
            continue
        for patch1_id in image2patch[image1]:
            for patch2_id in image2patch[image2]:
                affinity_matrix[patch1_id, patch2_id] = affinity_matrix[patch1_id, patch2_id] = patch_list[patch1_id].compute_similarity(patch_list[patch2_id])

    np.save(save_path, affinity_matrix)
    return affinity_matrix


def patch_co_vis_grouping_new(patch_list, image2patch_path, db_image_pairs_path, group_result_dir, n_cluster = 2000):
    patch_num = len(patch_list)
    
    affinity_matrix = patch_similarity_computation(patch_list, db_image_pairs_path, image2patch_path, "group/aachen_patch/a_matrix.npy")
    labels_path = group_result_dir / "cluster_result.pkl"
    
    distance_matrix = 1-affinity_matrix
    aggclus = AgglomerativeClustering(n_clusters=n_cluster, 
                                      memory= str(str(group_result_dir) + "/temp"), 
                                      linkage="average",
                                      affinity = "precomputed").fit(distance_matrix)
    labels = aggclus.fit_predict(distance_matrix)
    
    index2patch_id = dict()
    ind = 0
    for patch in tqdm(patch_list):
        index2patch_id[ind] = ind
        ind += 1
        
    with open(labels_path, "wb") as f:
        pickle.dump(labels, f)
    id_path = group_result_dir / "index2patch_id.pkl"
    with open(id_path, "wb") as f:
        pickle.dump(index2patch_id, f)
    return labels, index2patch_id

def patch_desc_grouping(patch_global_feature_path, patch_list, group_result_dir, n_cluster = 2000):
    db_desc = h5py.File(patch_global_feature_path, 'r')

    labels_path = group_result_dir / "cluster_result.pkl"
    id_path = group_result_dir / "index2patch_id.pkl"
    group_desc = []
    index2patch_id = dict()
    ind = 0
 
    for patch in tqdm(patch_list):
        group_desc.append(np.array(db_desc[str(ind) + '.png']['global_descriptor']))
        index2patch_id[ind] = ind
        ind += 1

    print(len(group_desc))
    group_desc=np.array(group_desc)
    
    aggclus = AgglomerativeClustering(n_clusters=n_cluster, memory=str(str(group_result_dir) + "/temp")).fit(group_desc)
    labels = aggclus.labels_
    
    with open(labels_path, "wb") as f:
        pickle.dump(labels, f)
    with open(id_path, "wb") as f:
        pickle.dump(index2patch_id, f)
    return labels, index2patch_id

#  patches match and merge
def co_vis_grouping(patch_list, patch_set_path, db_image_pairs_path, image2patch_path, image_dir,
                    do_crop = False, group_output_dir = None, images = None):
    
    if os.path.exists(patch_set_path):
        with open(patch_set_path, "rb") as f:
            patch_set = pickle.load(f)
    else:
        patch_set = PatchSet()

        for i in range(len(patch_list)):
            patch_set.fa.append(i)

        with open(image2patch_path, "rb") as f:
            image2patch = pickle.load(f)

        with open(db_image_pairs_path, "r") as f:
            db_image_pairs = f.readlines()

        logging.info("grouping...")

        for pair in tqdm(db_image_pairs):
            image1 = pair.split()[0]
            image2 = pair.split()[1]
            if image1 < image2:
                continue
            for patch1_id in image2patch[image1]:
                for patch2_id in image2patch[image2]:
                    if patch_set.find(patch2_id) != patch2_id:
                        continue
                    if (patch_list[patch1_id].check_match(patch_list[patch2_id]) and
                        patch_list[patch_set.find(patch1_id)].check_match(patch_list[patch2_id])):
                        patch_set.merge(patch1_id, patch2_id)        

        with open(patch_set_path, "wb") as f:
            pickle.dump(patch_set, f)
     
#     if do_crop:
#         logging.info("Cropping...")
#         for i in tqdm(range(len(patch_list))):
#             image_path = os.path.join(image_dir, images[patch_list[i].image_id].name)
#             save_path = os.path.join(group_output_dir, str(patch_set.find(i)), str(i)+ ".png")
#             patch_list[i].save_crop(image_path, save_path)
            
    return patch_set



def points_grouping(patch_list, images, labels, index2patch_id, patch_global_feature_path):
    patch_group_points = defaultdict(set) 
    group_descriptor = defaultdict(list) 
    
    db_desc = h5py.File(patch_global_feature_path, 'r')
    
    for i in tqdm(range(len(labels))):
        patch = patch_list[index2patch_id[i]]
        p3d_id_set = set(images[patch.image_id].point3D_ids)   
#         p3d_id_set = set(patch.p3d_ids)        
        
        patch_group_points[labels[i]] = patch_group_points[labels[i]].union(p3d_id_set) 
        group_descriptor[labels[i]].append(np.array(db_desc[str(i) + '.png']['global_descriptor']))
        
    for label, desc in tqdm(group_descriptor.items()):
        if len(desc) == 0:
            group_descriptor.pop(label, None)
            continue
        arr = np.array(desc)
        dist = cdist(arr, np.mean(arr, axis=0).reshape((1,4096)), metric='euclidean')
        dist_idx = np.argmin(dist, axis=0)
        group_descriptor[label] = arr[dist_idx].reshape(4096)
        
    cluster_point_num = []
    for gid, cps in patch_group_points.items():
        cluster_point_num.append(len(cps))
        
    somelist_df = pd.DataFrame(cluster_point_num)
    print(somelist_df.describe())
        
    return patch_group_points, group_descriptor


def build_tree(group_descriptor, nca = None):
    group_desc = []
    index2name = dict()
    ind = 0
    for gid, desc in tqdm(group_descriptor.items()):
        group_desc.append(desc)
        index2name[ind] = gid
        ind += 1

    if nca is not None:
        group_desc = nca.transform(group_desc)
        
    group_desc=np.array(group_desc)
    tree = spatial.KDTree(group_desc)
    return tree, index2name


def generate_retrieval_file(global_feature_path, tree, index2name, query_images, retrieval_dir,  
                            nca = None, label = "whole", top_n = 50):
    desc = h5py.File(global_feature_path, 'r')

    retrieval_path = retrieval_dir / str("retrieval_" + label + "_"+str(top_n)+".txt")
    file = open(retrieval_path, "w+")

    for query_image in tqdm(query_images):
        name, param = query_image
        
        cur_desc = np.array([desc[name]['global_descriptor']])
        if nca is not None:
            cur_desc = nca.transform(cur_desc)[0]
            
        _, group_indices = tree.query(cur_desc, top_n)
        
        if top_n == 1:
            file.write(name + ' ' + str(index2name[group_indices]) + '\n') 
        else:
            for i in range(len(group_indices)):
                file.write(name + ' ' + str(index2name[group_indices[i]]) + '\n')     

def db_patch_point_bind(groups_patch_points_path, groups_image_points_path, patch_set_path, 
                        image2patch_path, new_images, patch_list):
    groups_patch_points=defaultdict(set) # for retrieval result examine
    groups_image_points=defaultdict(set) # for localization 

    with open(image2patch_path, "rb") as f:
        image2patch = pickle.load(f)
        
    with open(patch_set_path, "rb") as f:
        patch_set = pickle.load(f)
        
    for image_id, image in tqdm(new_images.items()):
        patches = image2patch[image.name]
        p3d_id_set = set(image.point3D_ids)
        for patch_id in patches:
            group_id = patch_set.find(patch_id)
            groups_image_points[group_id] = groups_image_points[group_id].union(p3d_id_set) 
            groups_patch_points[group_id] = groups_patch_points[group_id].union(patch_list[patch_id].p3d_ids) 
            
    with open(groups_patch_points_path, "wb") as f:
        pickle.dump(groups_patch_points,f)
    with open(groups_image_points_path, "wb") as f:
        pickle.dump(groups_image_points,f)
        
        
        
def generate_group_descriptor(db_feature_path ):
    db_desc = h5py.File(db_feature_path, 'r')['db']

    class_list =[]
    database = []
    index = 0
    val_desc = dict()

    for group in tqdm(db_desc.keys()):
        group_desc = []
        for ele in db_desc[group].keys():
            patch_id = int(ele.split('.')[0])
            group_desc.append(np.array(db_desc[group][ele]['global_descriptor']))#.flatten())

        if len(group_desc) > 0:
            group_desc = np.array(group_desc)
            database.append(np.mean(group_desc, axis=0))
            class_list.append(group)
    database=np.array(database)
    print(database.shape)
    tree = spatial.KDTree(database)
    return tree, class_list



def group_retrival(global_feature_path, tree, class_list, k = 5, gt = None):
    
    test_desc = dict()
#     test_desc_file = h5py.File(global_feature_path, 'r')['query']
    test_desc_file = h5py.File(global_feature_path, 'r')
    
    for ele in tqdm(test_desc_file.keys()):
        patch_id = int(ele.split('.')[0])
        test_desc[patch_id]=np.array(test_desc_file[ele]['global_descriptor'])

    gt_posi = dict()
    ave_posi = 0
    recall = 0
    retrieve_groups = dict()
    for patch_id in tqdm(test_desc.keys()):
        d, group_indices = tree.query(test_desc[patch_id], k)
        if k == 1:
            group_ids = [int(class_list[group_indices])]
        else:
            group_ids = [int(class_list[i]) for i in group_indices]
        retrieve_groups[patch_id] = group_ids
        if not gt == None:  
            for i in range(k):
                gt_posi[patch_id] = -1
                if (group_ids[i] == gt[patch_id] or 
                    len(patch_list[patch_id].p3d_ids.intersection(groups_patch_points[group_ids[i]])) > 50):
                    gt_posi[patch_id] = i
                    ave_posi += i+1
                    recall += 1
                    break
                    
    if not gt == None:
        # report
        print("recall:", recall/len(gt))
        print("ave. Posi:", ave_posi/recall)
        
    return retrieve_groups



def localize(query_images, test_retrieve_groups, local_feature_path, point_desc_path, test_image2patch_path,
             groups_image_points, points3D_new,images_new,cameras_new, log_path):

    print("localizing:", log_path)

    with open(test_image2patch_path, "rb") as f:
        test_image2patch = pickle.load(f)

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
    
    f = h5py.File(local_feature_path, 'r')
    
#     with open(groups_image_points_path, "rb") as f:
#         groups_image_points = pickle.load(f)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)

#     point_desc_file = h5py.File(point_desc_path, 'a')

    result = dict()
    max_inlier_nums = dict()
    infos = dict()
    for query_image in tqdm(query_images):
        file_name, param = query_image
        image_name = file_name.split('/')[-1]
        if image_name in result:
            continue
        candidate_point_set = list()
        patch_ids = test_image2patch[image_name]

        # point set merging
        for patch_id in patch_ids:
            groups = test_retrieve_groups[patch_id]
            for group_id in groups:
                merged_flag = False
                for i in range(len(candidate_point_set)):
                    if len(candidate_point_set[i].intersection(groups_image_points[group_id])) > 100:
                        merged_flag = True
                        candidate_point_set[i] = candidate_point_set[i].union(groups_image_points[group_id])
                        break
                if not merged_flag:
                    candidate_point_set.append(groups_image_points[group_id])
        
        cluster_infos = []
        max_inlier_num = 0
        final_qvec = np.array([-0.02718695,-0.12122045,0.20194369,0.97148609])
        final_tvec = np.array([723.98874206,76.41164933,187.85558868])

        candidate_point_set = sorted(candidate_point_set, key=len, reverse=True)
        
        # fusion & matching
        for cps in candidate_point_set:
            cluster_info = {"#points": len(cps)}
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
            p2d_descs = f[file_name]['descriptors'].__array__().transpose()
            matches = flann.knnMatch(p2d_descs, p3d_descs, k=2)

            mp3d = list()
            mkpq = list()
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.8*n.distance:
                    mp3d.append(points3D_new[p3d_ids[m.trainIdx]].xyz)
                    mkpq.append(f[file_name]['keypoints'][m.queryIdx])

            mp3d = np.array(mp3d).reshape(-1, 3)
            mkpq = np.array(mkpq).reshape(-1, 2)
            mkpq += 0.5
            cluster_info["#matches"] = mp3d.shape[0]
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
                    
            cluster_infos.append(cluster_info)
        result[image_name] = (final_qvec, final_tvec)
        max_inlier_nums[image_name] = max_inlier_num
        
        infos[image_name] = {"max_inlier_nums":max_inlier_num, 
                             "#candidate" : len(candidate_point_set),
                             "candidates" : cluster_infos,
                             "max_inlier_nums": max_inlier_num
                            } 
        
    with open(log_path, 'w') as fp:
        json.dump(infos, fp)
#     f.close()
    del point_descs
#     point_desc_file.close()

    return result, max_inlier_nums