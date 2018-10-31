#!/bin/bash
# build file with all
wget https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/UNSW-NB15_1.csv -nc -O /tmp/UNSW-NB15_1.csv &
wget https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/UNSW-NB15_2.csv -nc -O /tmp/UNSW-NB15_2.csv &
wget https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/UNSW-NB15_4.csv -nc -O /tmp/UNSW-NB15_4.csv &
wget https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/UNSW-NB15_3.csv -nc -O /tmp/UNSW-NB15_3.csv

echo "srcip,sport,dstip,dsport,proto,state,dur,sbytes,dbytes,sttl,dttl,sloss,dloss,service,sload,dload,spkts,dpkts,swin,dwin,stcpb,dtcpb,smean,dmean,trans_depth,response_body_len,sjit,djit,Stime,Ltime,sinpkt,dinpkt,tcprtt,synack,ackdat,is_sm_ips_ports,ct_state_ttl,ct_flw_http_mthd,is_ftp_login,ct_ftp_cmd,ct_srv_src,ct_srv_dst,ct_dst_ltm,ct_src_ltm,ct_src_dport_ltm,ct_dst_sport_ltm,ct_dst_src_ltm,attack_cat,label" > UNSW-NB15_all.csv
cat /tmp/UNSW-NB15_1.csv >> UNSW-NB15_all.csv
cat /tmp/UNSW-NB15_2.csv >> UNSW-NB15_all.csv
cat /tmp/UNSW-NB15_3.csv >> UNSW-NB15_all.csv
cat /tmp/UNSW-NB15_4.csv >> UNSW-NB15_all.csv

wget https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/a%20part%20of%20training%20and%20testing%20set/UNSW_NB15_training-set.csv -O UNSW_NB15_testing-set.csv
wget https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/a%20part%20of%20training%20and%20testing%20set/UNSW_NB15_testing-set.csv -O UNSW_NB15_training-set.csv
