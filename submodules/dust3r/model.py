from .mast3r.model import AsymmetricMASt3R
from .mast3r.retrieval.processor import Retriever

from .dust3r.inference import inference
from .dust3r.model import AsymmetricCroCo3DStereo
from .dust3r.cloud_opt.pair_aligner import PairAligner
from .dust3r.cloud_opt.pair_viewer import  PairViewer

import torch
import numpy as np
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ckpt = 'submodules/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth'
ckpt = 'submodules/dust3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
# ckpt = 'submodules/dust3r/checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth'
retrieval_ckpt = "submodules/dust3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"

class Dust3r:
    def __init__(self, K, w, h, size):
        self.device = torch.device('cuda')
        self.dtype = torch.float32
        self.model = AsymmetricMASt3R.from_pretrained(ckpt).to(self.device)
        self.retriever = Retriever(retrieval_ckpt, backbone=self.model, device=self.device)
        self.current_retriever_id = 0
        self.retriever_id_dict = dict()
        # self.model = AsymmetricCroCo3DStereo.from_pretrained(ckpt).to(self.device)

        S = max(w, h)
        if S > size:
            self.interp = Image.LANCZOS
        elif S <= size:
            self.interp = Image.BICUBIC
        
        new_w = int(round(w*size/S))
        new_h = int(round(h*size/S))
        self.new_size = tuple([new_w, new_h])

        self.new_K = np.eye(3)
        self.new_K[0,:] = K[0,:]*float(self.new_size[0])/w
        self.new_K[1,:] = K[1,:]*float(self.new_size[1])/h

        mid_x = new_w // 2
        mid_y = new_h // 2
        halfw, halfh = ((2*mid_x)//16)*8, ((2*mid_y)//16)*8

        self.left = mid_x - halfw
        self.upper = mid_y - halfh
        self.right = mid_x + halfw
        self.lower = mid_y + halfh

        self.new_K[0, 2] -= self.left
        self.new_K[1, 2] -= self.upper

        self.new_w = halfw*2
        self.new_h = halfh*2

    @property
    def resized_K(self):
        return self.new_K
    
    @property
    def input_img_size(self):
        return self.new_w, self.new_h
    
    def preprocess(self, pil_img):
        resized_img = pil_img.resize(self.new_size, self.interp)
        croped_img = resized_img.crop((self.left, self.upper, self.right, self.lower))
        
        torch_img = (
            torch.from_numpy(np.array(croped_img) / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )

        return torch_img
    
    def preprocess_depth(self, depth):
        depth_map_resized = Image.fromarray(depth).resize(self.new_size, resample=Image.NEAREST)
        cropped_depth = np.array(depth_map_resized.crop((self.left, self.upper, self.right, self.lower)))
        cropped_depth = torch.from_numpy(cropped_depth).to(self.device).to(self.dtype)
        return cropped_depth
        

    def predict_2view(self, torch_img_pair, second_K=None):

        shape = np.int32([[self.new_h, self.new_w]])
        input_0 = dict(img=(2*torch_img_pair[0] - 1).unsqueeze(0), true_shape=shape, idx=0, instance=str(0))
        input_1 = dict(img=(2*torch_img_pair[1] - 1).unsqueeze(0), true_shape=shape, idx=1, instance=str(1))

        if second_K is None :
            pairs = [(input_0, input_1), (input_1, input_0)]
        else :
            pairs = [(input_0, input_1)]

        output = inference(pairs, self.model, self.device, batch_size=1, verbose=False)
        with torch.no_grad() :
            view1, pred1 = output['view1'], output['pred1']
            view2, pred2 = output['view2'], output['pred2']
        
        net = PairAligner(view1, view2, pred1, pred2).to(self.device)
        
        if second_K is None :
            net.align_points_bidirection()
        else :
            net.align_points(second_K)

        poses = net.get_im_poses()
        depths = net.get_depthmaps()
        confidences = net.get_confidences()
        Ks = net.get_intrinsics()

        return poses, depths, confidences, Ks
    
    def add_img_to_retriever(self, torch_img, uid):
        shape = np.int32([[self.new_h, self.new_w]])
        input_img = dict(img=(2*torch_img - 1).unsqueeze(0), true_shape=shape, idx=0, instance=str(0))
        with torch.no_grad():
            self.retriever.add(input_img, self.current_retriever_id)

        self.retriever_id_dict[self.current_retriever_id] = uid
        self.current_retriever_id += 1

    def query_from_retriever(self, torch_img, uid):
        if self.current_retriever_id == 0:
            return
        shape = np.int32([[self.new_h, self.new_w]])
        query_img = dict(img=(2*torch_img - 1).unsqueeze(0), true_shape=shape, idx=0, instance=str(0))
        with torch.no_grad():
            ranks, scores = self.retriever.query(query_img, uid)

        ids = [self.retriever_id_dict[i] for i in ranks[0]]
        return ids, scores[0]
            




        




