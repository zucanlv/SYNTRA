#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
查看TSV文件的前5行内容
TSV文件是制表符分隔的文本文件，与CSV类似但分隔符不同
"""

def read_tsv_first_n_lines(file_path, n=5):
    """
    读取TSV文件的前n行
    
    Args:
        file_path (str): 文件路径
        n (int): 要读取的行数，默认5行
    
    Returns:
        list: 包含前n行内容的列表
    """
    lines = []
    try:
        # 打开文件，指定编码（MSMARCO数据集通常用utf-8）
        with open(file_path, 'r', encoding='utf-8') as f:
            # 逐行读取，直到读取n行或文件结束
            for i, line in enumerate(f):
                if i >= n:
                    break
                # 去除行尾换行符，并按制表符分割字段
                cleaned_line = line.strip()
                lines.append(cleaned_line.split('\t'))
        
        # 打印结果
        print(f"文件 {file_path} 的前{n}行内容：")
        print("-" * 80)
        for idx, line in enumerate(lines, 1):
            print(f"第{idx}行:")
            # 打印每个字段（MSMARCO triples文件通常有3列：query, positive passage, negative passage）
            for col_idx, field in enumerate(line):
                print(f"  字段{col_idx+1}: {field}")
            print("-" * 80)
            
        return lines
    
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return []
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return []

# 主程序
if __name__ == "__main__":
    # 指定文件路径
    file_path = "/data/share/project/shared_datasets/MSMARCO/triples.train.small.tsv"
    # 读取前5行
    read_tsv_first_n_lines(file_path, n=100)


'''
文件 /data/share/project/shared_datasets/MSMARCO/triples.train.small.tsv 的前5行内容：
--------------------------------------------------------------------------------
第1行:
  字段1: is a little caffeine ok during pregnancy
  字段2: We donât know a lot about the effects of caffeine during pregnancy on you and your baby. So itâs best to limit the amount you get each day. If youâre pregnant, limit caffeine to 200 milligrams each day. This is about the amount in 1Â½ 8-ounce cups of coffee or one 12-ounce cup of coffee.
  字段3: It is generally safe for pregnant women to eat chocolate because studies have shown to prove certain benefits of eating chocolate during pregnancy. However, pregnant women should ensure their caffeine intake is below 200 mg per day.
--------------------------------------------------------------------------------
第2行:
  字段1: what fruit is native to australia
  字段2: Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.assiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.
  字段3: The kola nut is the fruit of the kola tree, a genus (Cola) of trees that are native to the tropical rainforests of Africa.
--------------------------------------------------------------------------------
第3行:
  字段1: how large is the canadian military
  字段2: The Canadian Armed Forces. 1  The first large-scale Canadian peacekeeping mission started in Egypt on November 24, 1956. 2  There are approximately 65,000 Regular Force and 25,000 reservist members in the Canadian military. 3  In Canada, August 9 is designated as National Peacekeepersâ Day.
  字段3: The Canadian Physician Health Institute (CPHI) is a national program created in 2012 as a collaboration between the Canadian Medical Association (CMA), the Canadian Medical Foundation (CMF) and the Provincial and Territorial Medical Associations (PTMAs).
--------------------------------------------------------------------------------
第4行:
  字段1: types of fruit trees
  字段2: Cherry. Cherry trees are found throughout the world. There are 40 or more varieties, ranging from bing cherry to black cherry. Along with the fruit, cherry trees produce light and delicate pinkish-white blossoms that are highly fragrant.omments. Submit. Planting fruit trees on your property not only provides you with a steady supply of organic fruit, it also allows you to beautify your yard and give oxygen back to the environment.
  字段3: The kola nut is the fruit of the kola tree, a genus (Cola) of trees that are native to the tropical rainforests of Africa.
--------------------------------------------------------------------------------
第5行:
  字段1: how many calories a day are lost breastfeeding
  字段2: Not only is breastfeeding better for the baby, however, research also says itâs better for the mother. Breastfeeding burns an average of 500 calories a day, with the typical range from 200 to 600 calories burned a day. Itâs estimated that the production of 1 oz. of breast milk burns 20 calories. The amount of calories burned depending on how much the baby eats. Breastfeeding twins burn twice as much as feeding only one baby. With twins their mom burns 1000 calories a day. Burning an extra 500 calories a day will result in one pound of weekly weight loss.
  字段3: However, you still need some niacin each day; men need about 16 mg per day and women need 14 mg per day unless they are pregnant or nursing (pregnant and breastfeeding women have higher niacin requirements).
--------------------------------------------------------------------------------
'''