# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SMT',
        embed_dims=[64, 128, 256, 512],
        ca_num_heads=[4, 4, 4, -1],
        sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[4, 4, 4, 2],
        qkv_bias=True,
        drop_path_rate=0.2,
        depths=[3, 4, 18, 2],
        ca_attentions=[1, 1, 1, 0],
        num_stages=4,
        head_conv=3,
        expand_ratio=2,
        init_cfg=dict(type='Pretrained', checkpoint='checkpoints/pretrained/smt_small.pth'),
        ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,  # Changed to 2 for binary segmentation
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),  # Changed use_sigmoid to True
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,  # Changed to 2 for binary segmentation
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),  # Changed use_sigmoid to True
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))