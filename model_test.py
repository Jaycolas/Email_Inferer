#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
import tensorflow as tf
from text_cnn import TextCNN
from util import create_or_load_hparams, load_obj

HFILE="./standard_hparams/default.json"
ROOT_PATH ='./dataset/jira'

TEST_EMAIL='''
From: Huimin Cao 
Sent: Friday, December 14, 2018 3:28 PM
To: Jory Zhou <jingz@qti.qualcomm.com>; Nan Shen (Jason) <nshen@qti.qualcomm.com>; Chaokai Xu <chaoxu@qti.qualcomm.com>; Zhe Wang <zhew@qti.qualcomm.com>; Gary Zhang <zhzheng@qti.qualcomm.com>
Cc: Weihang Zhao <c_weihzh@qti.qualcomm.com>; Fei Huang <huangf@qti.qualcomm.com>; Ruimin Wang <ruimwang@qti.qualcomm.com>
Subject: RE: TDS notch NV setting on Atlas

Hi Jory/Jason/Chaokai,
I have only configured notch filter NV on TA. I just remember on AT Xu Huan/Jory once worked for that part to do it from DTR side.. not sure whether processing NV for this on AT. 

On TA, we can config NV like this:
RFNV_TDSCDMA_C0_SPURS_TABLE_I                           = 25530,  (QXDM:71295)

Thanks,
Huimin


From: Jory Zhou 
Sent: Friday, December 14, 2018 10:54 AM
To: Nan Shen (Jason) <nshen@qti.qualcomm.com>; Chaokai Xu <chaoxu@qti.qualcomm.com>; Zhe Wang <zhew@qti.qualcomm.com>; Gary Zhang <zhzheng@qti.qualcomm.com>; Huimin Cao <huiminc@qti.qualcomm.com>
Cc: Weihang Zhao <c_weihzh@qti.qualcomm.com>; Fei Huang <huangf@qti.qualcomm.com>; Ruimin Wang <ruimwang@qti.qualcomm.com>
Subject: RE: TDS notch NV setting on Atlas

Below is what I collect from ATLAS. Not sure whether they are still valid now.

@Huimin, I voguely remember you ever added notch filter before. Could you remember how you configure the notch filter static NV in the past?

Have an API similar to rf_tdscdma_msm_update_dtr_notch_settings in rf_tdscdma_msm.c exposed to MC level , in current CL , it retrieve notch setting from NV in flow of rf_tdscdma_msm_update_dynamic_rxlm_buffer() , but as this rxlm buffer update API will be called per chain (PRX , DRX ) , and even in IFREQ or IRAT flow  , let have an API called rf_tdscdma_msm_retrieve_dtr_notch_settings() , the input is Rx NV item , inside this API it will process and retrieve notch setting in rflm_dtr_rx_notch_filter_cfg_t format , then store this out structure in rflmTdsrfmodectl variable. [rflm_tds_notch_cfg is already a global variable]
So we only call this API in enable_rx and update BHO these two case to update new notch setting after changing frequency 
Inside rflm_tds_dtr_rx_config_chain() API , we call DTR side rflm_dtr_rx_update_notch_filters() with stored notch setting , this step is to put notch setting into DTR’s “template” , so later in step of rflm_tds_dtr_rxlm_update() -> rflm_dtr_rx_commit_write_template_ag() , it will write the template into HW.
The key point to make notch setting effect is inside flow rflm_tds_dtr_rxlm_update() -> rflm_dtr_rx_start_notch_filters() , this enable start notch bit in template , after template written to HW , latch will start.
As we don’t want notch is being used in ifreq ( X2T , IRAT we could discuss later ) , in rflm_tds_dtr_rxlm_update() , there is a isIfreq field we could use , currently per my talk to Qilong, TFW doesn’t populate this field yet , but it should be fine for them to populate , his only concern is that which PL we might want for this feature. Some atlas target currently are very restrict for them to check-in.

Best regards,
Jory

From: Nan Shen (Jason) 
Sent: Friday, December 14, 2018 10:37 AM
To: Chaokai Xu <chaoxu@qti.qualcomm.com>; Zhe Wang <zhew@qti.qualcomm.com>; Gary Zhang <zhzheng@qti.qualcomm.com>
Cc: Weihang Zhao <c_weihzh@qti.qualcomm.com>; Fei Huang <huangf@qti.qualcomm.com>; Jory Zhou <jingz@qti.qualcomm.com>; Ruimin Wang <ruimwang@qti.qualcomm.com>
Subject: RE: TDS notch NV setting on Atlas

Hi, Chaokai

I am not quite familiar with the whole procedure.
Adding one more entry in rfnv_items.h and its associated handling code will be a must, but I am not sure what needs to be done in NV definition and other possible SS.
@Gary Zhang, Could you please chimme in?


--
Best Regards,
Nan Shen(Jason)



From: Chaokai Xu 
Sent: Friday, December 14, 2018 10:30 AM
To: Nan Shen (Jason) <nshen@qti.qualcomm.com>; Zhe Wang <zhew@qti.qualcomm.com>
Cc: Weihang Zhao <c_weihzh@qti.qualcomm.com>; Fei Huang <huangf@qti.qualcomm.com>; Jory Zhou <jingz@qti.qualcomm.com>; Ruimin Wang <ruimwang@qti.qualcomm.com>
Subject: RE: TDS notch NV setting on Atlas

Hi Jason,

Could you help to guide how to add TDS notch NV? 
I am not familiar with it. Does it means  add a new event in NV?

Best Regards,
Chaokai Xu.

'''

def model_test():

    hparams = create_or_load_hparams(hparams_file=HFILE, default_hparams=None)
    config = tf.ConfigProto(log_device_placement=hparams.log_device_placement,
                            allow_soft_placement=hparams.allow_soft_placement)
    input_vocab = load_obj(ROOT_PATH, 'general_vocab')
    label_vocab = load_obj(ROOT_PATH, 'mpss_pl_vocab')

    with tf.Session(config=config) as sess:
        cnn = TextCNN(hparams=hparams,
                      mode=tf.contrib.learn.ModeKeys.TRAIN,
                      source_vocab_table=input_vocab,
                      target_vocab_table=label_vocab,
                      scope=None,
                      extra_args=None)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=hparams.num_checkpoints)
        chpt = tf.train.latest_checkpoint(hparams.restore_checkpoint)
        if chpt:
            if tf.train.checkpoint_exists(chpt):
                saver.restore(sess, chpt)
                print("Model has been resotre from %s"%(hparams.restore_checkpoint))
        else:
            print("No pre-trained model loaded, abort!!!")
            return

        sess.run(tf.local_variables_initializer())

        predict_result = cnn.predict(sess=sess, input_txt=TEST_EMAIL, input_vocab=input_vocab, label_vocab=label_vocab)

        print("Predicted result is %s"%predict_result)


if __name__ == '__main__':
    model_test()
