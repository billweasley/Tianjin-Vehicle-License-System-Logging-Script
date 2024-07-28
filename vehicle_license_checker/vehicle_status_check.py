# -*- coding: utf-8 -*-

import datetime
import time
from typing import Dict, Tuple
from vehicle_license_checker.verifier_modeling import Verifier
from selenium import webdriver
from selenium.webdriver.common.by import By
import base64
import yaml
import argparse

LOGIN_URL = "https://xkctk.jtys.tj.gov.cn/"

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def base64_to_image(base64_string: str) -> Tuple[str, str]:

    type = "png" # default
    if "data:image" in base64_string:
        type, base64_string = base64_string.split(",")
        type = type.split("/")[1].split(";")[0]
    # Decode the Base64 string into bytes
    image_bytes = base64.b64decode(base64_string)
    return image_bytes, type

def get_info(args: Dict, verifer: Verifier) -> Dict:

    driver = webdriver.Chrome(args["selenium_chrome_driver_path"])
    driver.get(LOGIN_URL)

    initial_login = driver.find_element(By.CLASS_NAME, "login-reg")
    initial_login.click()
    driver.implicitly_wait(5)
    follow_login = driver.find_element(By.LINK_TEXT, "个人登录")
    follow_login.click()
    driver.implicitly_wait(5)

    username_input = driver.find_element(By.XPATH, '//input[@placeholder="用户名/手机号/证件号码/外国人永久居留身份证"]')
    password_input = driver.find_element(By.XPATH, '//input[@placeholder="请输入密码"]')
    # captcha_input = driver.find_element(By.XPATH, '//input[@placeholder="请输入图形验证码"]')
    captcha_img_base64 = driver.find_element(By.CLASS_NAME, "capche").find_element(By.XPATH, '//img').get_attribute('src')
    captcha_bytes, captcha_img_type = base64_to_image(captcha_img_base64)

    captcha_bytes_save_path = f"test.{captcha_img_type}"
    with open(captcha_bytes_save_path, "wb+") as write_handler:
        write_handler.write(captcha_bytes)
    
    # captcha_result = verifer.get_result(captcha_bytes_save_path)
    # captcha_input.send_keys(captcha_result)

    username_input.send_keys(args["phone"])
    password_input.send_keys(args["password"])
    time.sleep(20)

    login_button = driver.find_element(By.CLASS_NAME, "login-btn")
    login_button.click()
    driver.implicitly_wait(5)
   
    apply_status = driver.find_element(By.CLASS_NAME, 'el-table__row').find_element(By.CLASS_NAME, 'applyStatus-column').find_element(By.CLASS_NAME, 'cell').text
    apply_time  = driver.find_element(By.CLASS_NAME, 'el-table__row').find_element(By.CLASS_NAME, 'applyTime-column').find_element(By.CLASS_NAME, 'cell').text

    return {
        "手机": args["phone"],
        "请求时间": str(datetime.datetime.now()),
        "申请时间": apply_time,
        "申请状态": apply_status
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Read YAML configuration file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    config = read_config(config_path)
    verifer = Verifier(config["model_path"])
    print(get_info(config, verifer))