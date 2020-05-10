# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 15:38:43 2020

@author: HP
"""

import numpy as np
from numba import jit
import math
import cv2
import matplotlib.pyplot as plt
#from scipy import integrate 

def black_hole_simulation(img, L1 = 1, L2 = 1, M = 1, D = 1):

#    L1 = 5e21
#    L2 = 5e21
#    M = 1e19 
    @jit
    def solve_cubic(b):
    #    w = -0.5 + np.sqrt(3)/2*1j
        p = -b**2
        q = -2*M*b**2
        delta = q**2/4 + p**3/27
        
        if delta > 0:
            return False, 0
        Q = -p/3
        R = q/2
        temp = R/Q/np.sqrt(Q)
        if temp > 1:
            temp = 1
        elif temp < -1:
            temp = -1
        theta = np.arccos(temp)
        
            
        x = 2*np.sqrt(Q)*np.cos((theta + 0*np.pi)/3)
    #    y1 = -q/2 + np.sqrt(-delta)*1j
    #    y2 = -q/2 - np.sqrt(-delta)*1j
    ##    x = np.cbrt(y1) + np.cbrt(y2)
    #    x = 2 * (np.cbrt(abs(y1)) * cmath.exp(1j*1*cmath.phase(y1)/3) * w).real
    
        if math.isnan(x) or math.isinf(x):
            return False, 0
    
        return True, x
    
    
    @jit
    def derivative_func(x, b, R, deriv_b = False):
        
        temp = R**2 / b**2 - x**2 + 2*M / R * x**3
        if temp > 0:
            return 1 / np.sqrt(temp)
        else:
            return 1/ 10**(-12)
        
    
    def show_derivative():
        b = 100
        ret, R = solve_cubic(b)
        x = np.linspace(0, 1-1e-10, 500)
        print(R)
        y = []
        for item in x:
    #        y.append(solve_cubic(item)[1])
            y.append(derivative_func(item, b, R, True))
    #    y = derivative_func(x, b, False)
        plt.figure()
        plt.plot(x, y)
    #show_derivative()
    #m=n
    
    @jit
    def seg_integrate(start, b, R, deriv_b = False):
    #    print('b',b, 'R',R)
        seg = [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 1-1e-6, 1-1e-7, 1-1e-8, 
               1-1e-9, 1-1e-10]
        
        delta_x = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 
                   1e-12, 1e-13]
        x = start
        y = 0
        
        for i in range(len(seg)):
            if x < seg[i]:
                index = i
                break
    #    print('123')
        while index <= len(seg) - 1:
            x += delta_x[index]
            while x < seg[index]:
                delta_y = delta_x[index] * (derivative_func(x, b, R, deriv_b) + 
                                            derivative_func(x - delta_x[index], b, R, deriv_b)) / 2
                y += delta_y
                x += delta_x[index]
            x -= delta_x[index]
            index += 1
    #    print('456')
        return y
            
        
    
    @jit
    def integrate(b, R, d, deriv_b = False):
        angle = 0
        
        angle += seg_integrate(R/np.sqrt(L1**2 + d**2), b, R, deriv_b)
    #    print('1',angle)
        angle += seg_integrate(R/L2, b, R, deriv_b)
    #    print('2',angle)
    #    print('789')
    #    print(angle)
        if not deriv_b:
            angle_1 = np.arctan(L2 / L2**2 / np.sqrt(1/b**2 - 1/L2**2 + 2*M/L2**3))
            return angle, angle_1
        else:
            return angle, 0
    
    @jit
    def solve_derivatives(b, R, d):
        b = max(np.sqrt(27)*M + 10**-2, b)
    #    ret, R = solve_cubic(b)
    #    angle, _ = integrate(b, R, d, True)
    #    print(b)
    #    return angle
        n = 4
        points = np.zeros(n + 1)
        b_list = np.zeros(n + 1)
        delta = 0.1
        b_temp = b - delta * n/2
        for i in range(n + 1):
            b = max(np.sqrt(27)*M+10**-6, b_temp)
            ret, R = solve_cubic(b)
            b_list[i] = b
            angle, _ = integrate(b, R, d)
            points[i] = angle
            b_temp += delta
        k = ((n+1) * np.sum(b_list*points) - np.sum(b_list) * np.sum(points)) / \
            (((n+1) * np.sum(b_list**2) - (np.sum(b_list))**2))
        if ((n+1) * np.sum(b_list**2) - (np.sum(b_list))**2) == 0:    
            print(((n+1) * np.sum(b_list**2) - (np.sum(b_list))**2), b_list)
            print(k)
    #    print(k)
        return k
    
    
    def find_b_newton(target, d):
        b = np.sqrt(27) * M + 0.1
        ret, R = solve_cubic(b)
        while not ret:
            b += 0.1
            ret, R = solve_cubic(b)
        eps = 10**(-7)
        count = 0
        angle, angle_1 = integrate(b, R, d)
        while abs(angle - target) > eps:
             b = b - (angle - target)/solve_derivatives(b, R, d)
             ret, R = solve_cubic(b)
             angle, angle_1 = integrate(b, R, d)
             count = count + 1
    #         print(angle)
    #         print('count', count)
             if count > 20:
                 break
        return angle, angle_1
    
    @jit
    def find_b_2fen(target, d):
        b = np.sqrt(27) * M + 0.1
        ret, R = solve_cubic(b)
        count_find_initial_b = 0
        limit_find_initial_b = 200
        while not ret:
            b *= 1.01
            ret, R = solve_cubic(b)
            if count_find_initial_b > limit_find_initial_b:
                return -1, -1
        eps = 10**(-7)
        count = 0
        limit_count = 200
        delta_b = height / 10
        angle, angle_1 = integrate(b, R, d)
        for i in range(100):
            if angle < target:
                break
            b += delta_b
            ret, R = solve_cubic(b)
            angle, angle_1 = integrate(b, R, d)
        
        while count < limit_count:
            delta_b /= 2
            while angle < target:
                b -= delta_b
                ret, R = solve_cubic(b)
    #            print('b',b, 'R',R, 'delta', delta_b)
                if not ret:
                    b = np.sqrt(27) * M + 0.1
                    ret, R = solve_cubic(b)
                    while not ret:
                        b *= 1.01
                        ret, R = solve_cubic(b)
                angle, angle_1 = integrate(b, R, d)
                count += 1
                if abs(angle - target) < eps or count > limit_count:
                    break
                
            delta_b /=2
            while angle > target:
                b += delta_b
                ret, R = solve_cubic(b)
    #            print('b',b, 'R',R)
                angle, angle_1 = integrate(b, R, d)
                count += 1
                
                if abs(angle - target) < eps or count > limit_count:
                    break
        return angle, angle_1
            
            
            
    
    
    @jit
    def cal_alpha(d = 0):
        target_phi_1 = np.pi - np.arctan(d/L1)
        target_phi_2 = np.pi + np.arctan(d/L1)
        
        angle = 0
    #    angle, angle_1 = find_b_newton(target_phi_1, d)
        if abs(angle - target_phi_1) > 10**(-5):
            angle, angle_1 = find_b_2fen(target_phi_1, d)
            if abs(angle - target_phi_1) > 10**(-5):
                phi_1, alpha_1 = -1, -1
            else:
                phi_1, alpha_1 = angle, angle_1
        else:
            phi_1, alpha_1 = angle, angle_1
        
    #    angle, angle_1 = find_b_newton(target_phi_2, d)
        if abs(angle - target_phi_2) > 10**(-5):
            angle, angle_1 = find_b_2fen(target_phi_2, d)
            
            if abs(angle - target_phi_2) > 10**(-5):
                phi_2, alpha_2 = -1, -1
            else:
                phi_2, alpha_2 = angle, angle_1
        else:
            phi_2, alpha_2 = angle, angle_1
    #    print(b)
        print(phi_1 - target_phi_1, phi_2 - target_phi_2)
        print(alpha_1, alpha_2)
        
        return alpha_1, alpha_2
            
    
#    img = cv2.imread('F:/Desktop2020.1.17/BlackHole/161report/MilkyWay.jpg')
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.pyrUp(img)
    img = cv2.pyrUp(img)
    img = cv2.pyrUp(img)
    #img = cv2.pyrUp(img)
    #img = cv2.pyrUp(img)
#    cv2.imshow('1',img)
    height = D
    unit_len = height / img.shape[0]
    width = img.shape[1] * unit_len
    delta_d = height/img.shape[0]*50
    
    
#    @jit
    def find_alpha():
        d = 0
        d_list = [0]
        alpha_1_list = [0]
        alpha_2_list = [0]
    #    for i in range(1):
        for i in range(int(np.sqrt(img.shape[0]**2 + img.shape[1]**2)/2/50) + 3):
            alpha_1, alpha_2 = cal_alpha(d)
            alpha_1_list.append(alpha_1)
            alpha_2_list.append(alpha_2)
            d_list.append(d)
            d += delta_d
            
            print(i)
            
        alpha_1_list = alpha_1_list[1:]
        alpha_2_list = alpha_2_list[1:]
        d_list = d_list[1:]
        return d_list, alpha_1_list, alpha_2_list
    
    d_list, alpha_1_list, alpha_2_list = find_alpha()
    
#    plt.figure()
#    plt.plot(d_list, alpha_1_list)
#    plt.plot(d_list, alpha_2_list)
    
    #a = p1
    
    d_list_1 = d_list.copy()
    d_list_2 = d_list.copy()
    i = 0
    while i < len(d_list_1):
        if alpha_1_list[i] == -1:
            alpha_1_list.pop(i)
            d_list_1.pop(i)
            i -= 1
        if i == len(d_list_1):
            break
        i += 1
    i = 0
    while i < len(d_list_2):
        if alpha_2_list[i] == -1:
            alpha_2_list.pop(i)
            d_list_2.pop(i)
            i -= 1
        if i == len(d_list_2):
            break
        i += 1
    
    if len(alpha_1_list) == 0 or len(alpha_2_list) == 0:
        return np.zeros((512, 512, 3), dtype = np.uint8)
    for a, b in zip(alpha_1_list, alpha_2_list):
        if math.isnan(a) or math.isinf(a) or math.isnan(b) or math.isinf(b):
            return np.zeros((512, 512, 3), dtype = np.uint8)
    
    
    alpha_1_list = np.interp(d_list, d_list_1, alpha_1_list)
    alpha_2_list = np.interp(d_list, d_list_2, alpha_2_list)
    d_list = np.array(d_list)
    alpha_1_list = np.array(alpha_1_list)
    alpha_2_list = np.array(alpha_2_list)
#    plt.figure()
#    plt.plot(d_list, alpha_1_list)
#    plt.plot(d_list, alpha_2_list)
    
    
    
    max_height = 2 * np.tan(np.interp(height/2, d_list, alpha_1_list)) * (L1 + L2) / unit_len
    max_width = 2 * np.tan(np.interp(width/2, d_list, alpha_1_list)) * (L1 + L2) / unit_len
    
    zoom_out_rate = 1000/ max_height
    max_height *= zoom_out_rate
    max_width *= zoom_out_rate
    
    max_height += 10
    max_width += 10
    
    max_height = int(max_height)
    max_width = int(max_width)
    #a = p1
    @jit
    def gene_new_img():
        img_new = np.zeros((int(max_height), int(max_width), 3, 2))
        center_x_new = img_new.shape[1] / 2
        center_y_new = img_new.shape[0] / 2
        
        center_x = img.shape[1] / 2
        center_y = img.shape[0] / 2
        
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                x = (i - center_x + 1) * unit_len
                y = (j - center_y + 1) * unit_len
                r = np.sqrt(x**2 + y**2) 
                
                alpha_1 = np.interp(r, d_list, alpha_1_list)
                alpha_2 = np.interp(r, d_list, alpha_2_list)
                
                r_1 = np.tan(alpha_1) * (L1 + L2) / unit_len * zoom_out_rate
                r_2 = np.tan(alpha_2) * (L1 + L2) / unit_len * zoom_out_rate
                
                if r > 0:
                    x_1 = int(round(r_1 * x/r + center_x_new))
                    y_1 = int(round(r_1 * y/r + center_y_new))
                    
                    x_2 = int(round(r_2 * -x/r + center_x_new))
                    y_2 = int(round(r_2 * -y/r + center_y_new))
                    
                for k in range(3):
                    if y_1 < img_new.shape[0] and x_1 < img_new.shape[1]:
                        img_new[y_1, x_1, k, 0] += img[j, i, k]
                        img_new[y_1, x_1, k, 1] += 1
                    if y_2 < img_new.shape[0] and x_2 < img_new.shape[1]:
                        img_new[y_2, x_2, k, 0] += img[j, i, k]
                        img_new[y_2, x_2, k, 1] += 1
                
        for i in range(img_new.shape[0]):
            for j in range(img_new.shape[1]):
                for k in range(3):
                    if img_new[i, j, k, 1] > 0:
                        img_new[i, j, k, 0] /= img_new[i, j, k, 1]
        img_new = img_new[:,:,:,0].astype(np.uint8)
        
        alpha_min = alpha_1_list[0] * 0.8
        alpha_max = alpha_1_list[0] * 1.2
        r_min = np.tan(alpha_min) * (L1 + L2) / unit_len * zoom_out_rate
        r_max = np.tan(alpha_max) * (L1 + L2) / unit_len * zoom_out_rate
        print(r_min, r_max)
        nothing = np.zeros(3)
        for i in range(1, img_new.shape[0] - 1):
            for j in range(1, img_new.shape[1] - 1):
                x = (i - center_y_new + 1)
                y = (j - center_x_new + 1)
                r = np.sqrt(x**2 + y**2)
                
                if r_min <= r <= r_max:
                    if (img_new[i, j] == nothing).all():
                        temp = np.zeros(3)
                        count = 0
                        for m in range(-1, 2):
                            for n in range(-1, 2):
                                if not (img_new[i+m, j+n] == nothing).all():
                                    temp += img_new[i+m, j+n]
                                    count += 1
                        if count == 0:
                            continue
                        temp /= count
                        img_new[i, j] = temp.astype(np.uint8)
                        
        return img_new
    
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    img_new = gene_new_img()
    img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)
#    plt.figure()
#    plt.imshow(img_new, cmap = plt.cm.gray)
    #cv2.imshow('image', img_new) 
    #show_derivative()
    
    return img_new

if __name__ == '__main__':
    a = 0
    L1 = 10000
    L2 = 10000
    M = 1477
    D = 5000
    img = cv2.imread('F:/Desktop2020.1.17/BlackHole/161report/Sun.jpg')
    black_hole_simulation(img, L1, L2, M, D)
    
    
    
