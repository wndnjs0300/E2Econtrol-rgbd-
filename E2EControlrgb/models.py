import torch
import torch.nn as nn
import torch.nn.functional as F
#from utils_2d import PointNetSetAbstraction#, PointNetSetAbstractionMsg 
from torchvision import models
import timm

from einops.layers.torch import Rearrange
from transformer import Transformer

class net_vit_from_scratch(nn.Module):
    def __init__(
        self,        
        image_width = 256,
        image_height = 256,
        image_channels = 3,
        patch_size = 16,
        num_layers = 12,
        num_heads = 12,
        qkv_bias = True,
        mlp_ratio = 4.0,
        use_revised_ffn=False,
        attn_dropout_rate = 0.0,
        dropout_rate = 0.0,        
        embedding_dim = 768,        
        feat_dim_regression_head = 128,
        pose_dim = 4,
        position_embedding_dropout=None
    ):
        super(net_vit_from_scratch, self).__init__()
        
        patch_height, patch_width = patch_size, patch_size        
        num_patches = int((image_width * image_height) / (patch_size * patch_size))
        
        patch_dim = patch_height * patch_width * image_channels        
        self.embedding_dim = embedding_dim

        self.projection = nn.Sequential(
                            Rearrange(
                                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                p1=patch_height,
                                p2=patch_width,
                            ),
                            nn.Linear(patch_dim, self.embedding_dim)
                            )

        # class token for pose regression
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        num_patches += 1        
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.embedding_dim)
        )
        self.pe_dropout = position_embedding_dropout
        if self.pe_dropout is not None:
            self.pos_drop = nn.Dropout(p=self.pe_dropout)
        
        # transformer
        self.transformer = Transformer(
            dim=embedding_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout_rate,
            dropout=dropout_rate,
            qkv_bias=qkv_bias,
            revised=use_revised_ffn,
        )

        self.post_transformer_ln = nn.LayerNorm(self.embedding_dim)

        # regression layers
        self.to_cls_token = nn.Identity()
        self.post_dense = nn.Linear(self.embedding_dim, feat_dim_regression_head)
        self.post_relu = nn.ReLU()

        if pose_dim == 6:
            self.fc_t = nn.Linear(feat_dim_regression_head,3)
            self.fc_r = nn.Linear(feat_dim_regression_head,3)
        elif pose_dim == 4:
            self.fc_t = nn.Linear(feat_dim_regression_head,2)
            self.fc_r = nn.Linear(feat_dim_regression_head,2)                    


    def forward(self, img):                
        x = self.projection(img)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)        

        if self.pe_dropout is None:
            x = x + self.pos_embed
        else:
            self.pos_drop(x + self.pos_embed)

        x = self.transformer(x)        
        
        x = self.post_transformer_ln(x)
        x = self.to_cls_token(x[:, 0])

        x = self.post_relu(self.post_dense(x))

        pose_t = self.fc_t(x)
        pose_r = self.fc_r(x)

        pose = torch.cat([pose_t, pose_r], dim=1)

        return pose

class net_vit_pretrain(nn.Module):
    def __init__(self, feat_dim_regression_head = 128, cmdvel_dim = 2):
        super(net_vit_pretrain, self).__init__()
        
        # self.dropout_rate = dropout_rate
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        # self.model = timm.create_model('vit_medium_patch16_gap_256', pretrained=True)

        self.pos_embed = self.model.pos_embed
        self.cls_token = self.model.cls_token
        self.num_blocks = 12
        self.patch_embed = self.model.patch_embed
        self.pos_drop = self.model.pos_drop
        self.embed_dim = self.model.embed_dim
        self.feat_dim_regression_head = feat_dim_regression_head
        
        self.relu = nn.ReLU()

        self.head_MLP = nn.Linear(self.embed_dim, self.feat_dim_regression_head)
        self.head_relu = nn.ReLU()

        self.final_fc = nn.Linear(feat_dim_regression_head, cmdvel_dim)        

    def forward(self, x):

        # Divide input image into patch embeddings and add position embeddings        
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # Feed forward through transformer blocks
        for i in range(self.num_blocks):
            x = self.model.blocks[i](x)
        x = self.model.norm(x)
        
        # extract the cls token        
        cls_token_out = x[:, 0]
        x = self.head_relu(self.head_MLP(self.relu(cls_token_out)))

        cmdvel = self.final_fc(x)                
        
        return cmdvel

if __name__ == "__main__":

    #data = torch.rand(8,3,256,256)    
    data = torch.rand(8,3,224,224)

    model = net_vit_pretrain()
    # model = net_resnet_transformer(
    #     num_resnet_layers = 3,
    #     resnet_output_width = 128,
    #     resnet_output_height = 128,
    #     resnet_output_channels = 64,        
    #     patch_size = 8
    # )

    # model = net_resnet_transformer(
    #     num_resnet_layers = 5,
    #     resnet_output_width = 56,
    #     resnet_output_height = 56,
    #     resnet_output_channels = 64,        
    #     patch_size = 8
    # )

    # model = net_resnet_transformer(
    #     num_resnet_layers = 6,
    #     resnet_output_width = 28,
    #     resnet_output_height = 28,
    #     resnet_output_channels = 128,        
    #     patch_size = 4
    # )

    # model = net_resnet_transformer(
    #     num_resnet_layers = 7,
    #     resnet_output_width = 14,
    #     resnet_output_height = 14,
    #     resnet_output_channels = 256,        
    #     patch_size = 2
    # )

    # model = net_resnet_transformer(
    #     num_resnet_layers = 8,
    #     resnet_output_width = 7,
    #     resnet_output_height = 7,
    #     resnet_output_channels = 512,        
    #     patch_size = 1
    # )
    

    # print(model.__dict__.keys())
    pose = model(data)
    print(pose.shape)
    print(pose)    