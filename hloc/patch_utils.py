import numpy as np

from descartes import PolygonPatch
import alphashape

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import os

from PIL import Image, ImageDraw

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
#         print(rate, self.alphashape_thres)
        if rate >= self.alphashape_thres:
            self.validaty = True
        else:
            self.validaty = False
            
#         draw_points(image_path, np.array(whole_xys))
        return self.validaty

        # return self.validaty, p2d_id_inside
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
#         self.draw_points(image_path, self.p2d_xys.values())
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        im1.save(save_path)

    def check_coverage(self, p3d_ids, show = False):
        xys = [self.p2d_xys.get(key) for key in p3d_ids]
        AS = alphashape.alphashape(xys,5)  
        if show:
            print(AS.area, self.area, self.image_id)
            fig, ax = plt.subplots()
            # Plot input points
            ax.scatter(*zip(*xys))
            # Plot alpha shape
            ax.add_patch(PolygonPatch(AS, alpha=.2))
            ax.set_ylim(self.box[1],self.box[3])
            ax.set_xlim(self.box[0],self.box[2])
            
            plt.savefig("alphashape.png")
            
        return AS.area/self.area
    
    
def test_generate_patches(images, do_croping = False, crop_output_dir = None):
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
    return patch_list, image2patch




def database_generate_patches(images, image2patch_path):
    patch_list = list()
    image2patch = dict()
    
    # for error reporting
    no_box_detected_images = list()
    no_box_left_images = list()
    
    patch_cnt = 0
    
    for image_id, image in tqdm(images.items()):

        box_file_name = image.name.split('.')[0] + ".npy"

        image2patch[image_id] = list()

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
            logging.warning("no bbox found for "+ os.path.join(box_dir, box_file_name))
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
    
    return patch_list


#  patches match and merge
def co_vis_grouping(patch_list, patch_set_path, db_image_pairs_path, image2patch_path, 
                    do_crop = False, group_output_dir = None, images = None):
    
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
        patches1 = image2patch[image1]
        patches2 = image2patch[image1]
        
        for patch1_id in patches1:
            for patch2_id in patches2:
                if patch_set.find(patch2_id) != patch2_id:
                    continue
                if (patch_list[patch1_id].check_match(patch_list[patch2_id]) and
                    patch_list[patch_set.find(patch1_id)].check_match(patch_list[patch2_id])):
                    patch_set.merge(patch1_id, patch2_id)        
    
    with open(patch_set_path, "wb") as f:
        pickle.dump(patch_set, f)
     
    if do_crop:
        logging.info("Cropping...")
        for i in tqdm(range(len(patch_list))):
            image_path = os.path.join(image_dir, images[patch_list[i].image_id].name)
            save_path = os.path.join(group_output_dir, str(patch_set.find(i)), str(i)+ ".png")
            patch_list[i].save_crop(image_path, save_path)
            
    return patch_set



def db_patch_point_bind(groups_patch_points_path, groups_image_points_path, 
                        image2patch_path, new_images, patch_list):
    groups_patch_points=defaultdict(set) # for retrieval result examine
    groups_image_points=defaultdict(set) # for localization 

    with open(image2patch_path, "wb") as f:
        image2patch = pickle.load(f)
        
    for image_id, image in tqdm(new_images.items()):
        patches = image2patch[image_id]
        p3d_id_set = set(new_images.point3D_ids)
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
return tree



def group_retrival(test_desc, tree, k = 5, gt = None):
    gt_posi = dict()
    ave_posi = 0
    recall = 0
    retrieve_groups = dict()
    for patch_id in tqdm(test_desc.keys()):
        d, group_indices = tree.query(test_desc[patch_id], k)
        group_ids = [int(class_list[i]) for i in group_indices]
        retrieve_groups[patch_id] = group_ids
#         print(patch_id, group_ids, gt[patch_id])
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



def localize(query_images, test_retrieve_groups, local_feature_path, point_desc_path, test_image2patch_path, k = 5)


    with open(test_image2patch_path, "wb") as f:
        test_image2patch = pickle.load(f)

    f = h5py.File(local_feature_path, 'r')

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = k)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    point_desc_file = h5py.File(point_desc_path, 'a')

    result = dict()
    max_inlier_nums = dict()
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
    #                 print(len(groups_image_points[group_id]))
                    if len(candidate_point_set[i].intersection(groups_image_points[group_id])) > 100:
                        merged_flag = True
                        candidate_point_set[i] = candidate_point_set[i].union(groups_image_points[group_id])
                        break
                if not merged_flag:
    #                 print(len(groups_image_points[group_id]))
                    candidate_point_set.append(groups_image_points[group_id])

    #     print("# candidate_point_set: ", len(candidate_point_set))

        max_inlier_num = 0
        final_qvec = np.array([-0.02718695,-0.12122045,0.20194369,0.97148609])
        final_tvec = np.array([723.98874206,76.41164933,187.85558868])

        # fusion & matching
        for cps in candidate_point_set:
            p3d_descs = list()
            p3d_ids = list()
    #         print(len(cps))
            for p3d_id in cps:
                if p3d_id == -1:
                    continue
                p3d_ids.append(p3d_id)
                sid = str(p3d_id)
                if sid not in point_desc_file:
                    p3d=points3D_new[p3d_id]
                    desc = f[images_new[p3d.image_ids[0]].name]['descriptors'][:,p3d.point2D_idxs[0]]
                    point_desc_file.create_dataset(sid, data=desc)
    #                 print(sid)
                p3d_descs.append(point_desc_file[sid])

            p3d_descs = np.array(p3d_descs)
    #         print(p3d_descs.shape)
            tree = spatial.KDTree(p3d_descs)

            p2d_descs = f[file_name]['descriptors'].__array__().transpose()

            matches = flann.knnMatch(p2d_descs, p3d_descs, k=2)

    #         print(len(matches))
            mp3d = list()
            mkpq = list()
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    mp3d.append(points3D_new[p3d_ids[m.trainIdx]].xyz)
                    mkpq.append(f[file_name]['keypoints'][m.queryIdx])

            mp3d = np.array(mp3d).reshape(-1, 3)
            mkpq = np.array(mkpq).reshape(-1, 2)
            mkpq += 0.5

    #         camera = cameras_new[images[val_image_id].camera_id]
            cfg = {
                'model': param[0],
                'width': param[1],
                'height': param[2],
                'params': param[3],
            }
            ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg)
            if ret['success'] == True:
    #             print(ret['num_inliers'], ret['qvec'], ret['tvec'])
                if ret['num_inliers'] > max_inlier_num:
                    max_inlier_num = ret['num_inliers']
                    final_qvec = ret['qvec']
                    final_tvec = ret['tvec']
    #     print(max_inlier_num, final_qvec, final_tvec)
        result[image_name] = (final_qvec, final_tvec)
        max_inlier_nums[image_name] = max_inlier_num

    f.close()
    point_desc_file.close()

return result, max_inlier_nums