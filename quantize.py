#!/usr/bin/python
#coding:utf-8
from os import path
import argparse
from mrt.V3.utils import get_cfg_defaults, merge_cfg
from mrt.V3.execute import run

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_dir",    type=str,   default="./qconf", help="yaml config file location")
    parser.add_argument("--model",       type=str,   default="yolov5s", help="model name")
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    yaml_file = path.join(opt.conf_dir, opt.model+".yaml")
    cfg = get_cfg_defaults()
    cfg = merge_cfg(yaml_file)
    run(cfg)