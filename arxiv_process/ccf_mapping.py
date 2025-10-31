import requests
from bs4 import BeautifulSoup
import json

def normalize(text: str) -> str:
    """
    对文本进行归一化：去除首尾空格和换行，并转换为大写。
    """
    return text.strip().replace("\n", "").upper()

def scrape_ccf_list(url: str = "https://ccf.atom.im/") -> list:
    # 请求页面并解析
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # 定位到所有的 div.hideable 区块
    sections = soup.find_all('div', class_='hideable')
    if not sections:
        raise ValueError("页面中未找到任何 div.hideable 区块")

    data = []
    for idx, section in enumerate(sections, start=1):
        tbody = section.find('tbody')
        if not tbody:
            print(f"第 {idx} 个 hideable 区块中未找到 tbody，跳过。")
            continue

        for tr in tbody.find_all('tr', class_='item'):
            tds = tr.find_all('td')
            if len(tds) < 3:
                continue

            # 简称
            short_raw = tds[0].get_text()
            # 全称
            a_tag = tds[1].find('a')
            full_raw = a_tag.get_text() if a_tag else tds[1].get_text()
            # 分类
            category_raw = tds[2].get_text()

            # 归一化
            short_name = normalize(short_raw)
            full_name = normalize(full_raw)
            category = f"CCF {normalize(category_raw)}"

            data.append({
                "short_name": short_name,
                "full_name": full_name,
                "category": category
            })

    return data


def interactive_query(mapping: list):
    print("\n进入交互式查询模式。输入会议简称或全称，输入 'exit' 退出。\n")
    while True:
        query = input("请输入会议简称或全称：").strip()
        if query.lower() == 'exit':
            print("退出查询。")
            break
        query_norm = normalize(query)
        results = [entry for entry in mapping if query_norm in entry['short_name'] or query_norm in entry['full_name']]
        if results:
            for result in results:
                print(f"简称: {result['short_name']}, 全称: {result['full_name']}, 分类: {result['category']}")
        else:
            print("未找到匹配项。")

def main():
    """
    主函数：抓取 CCF 列表并输出到 JSON 文件。
    """
    mapping = scrape_ccf_list()
    output_path = 'arxiv_process_outputs\\ccf_mapping.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"已生成 {output_path}（共 {len(mapping)} 条记录）")
    # 进入交互式查询
    interactive_query(mapping)

if __name__ == '__main__':
    main()