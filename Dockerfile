FROM python:3.6-stretch

# add data
# files downloaded from  https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/
ADD UNSW-NB15_1.csv /tmp/UNSW-NB15_1.csv
ADD UNSW-NB15_2.csv /tmp/UNSW-NB15_2.csv
ADD UNSW-NB15_3.csv /tmp/UNSW-NB15_3.csv
ADD UNSW-NB15_4.csv /tmp/UNSW-NB15_4.csv

# training/testing filenames are switched (training should be the largest file)
ADD UNSW_NB15_testing-set.csv UNSW_NB15_testing-set.csv
ADD UNSW_NB15_training-set.csv UNSW_NB15_training-set.csv

RUN echo "srcip,sport,dstip,dsport,proto,state,dur,sbytes,dbytes,sttl,dttl,sloss,dloss,service,sload,dload,spkts,dpkts,swin,dwin,stcpb,dtcpb,smean,dmean,trans_depth,response_body_len,sjit,djit,Stime,Ltime,sinpkt,   dinpkt,tcprtt,synack,ackdat,is_sm_ips_ports,ct_state_ttl,ct_flw_http_mthd,is_ftp_login,ct_ftp_cmd,ct_srv_src,ct_srv_dst,ct_dst_ltm,ct_src_ltm,ct_src_dport_ltm,ct_dst_sport_ltm,ct_dst_src_ltm,attack_cat,label" > UNSW-NB15_all.csv && \
     cat /tmp/UNSW-NB15_1.csv >> UNSW-NB15_all.csv && \
     cat /tmp/UNSW-NB15_2.csv >> UNSW-NB15_all.csv && \
     cat /tmp/UNSW-NB15_3.csv >> UNSW-NB15_all.csv && \
     cat /tmp/UNSW-NB15_4.csv >> UNSW-NB15_all.csv

# install dependencies
RUN pip3 install tensorflow keras tabulate sklearn pandas ipython sklearn-deap bokeh==0.13.0 torch torchvision

# add code
ADD *.py ./
ADD bokeh_stream/ bokeh_stream/
ADD entrypoint.sh .

ENTRYPOINT ["/entrypoint.sh"]
