import argparse
import time
import arxiv
import pandas as pd
import requests
import os
import json
from bs4 import BeautifulSoup
import html
from tqdm import tqdm
import dotenv

# 加载期刊等级映射
with open('arxiv_process_outputs\\ccf_mapping.json', 'r', encoding='utf-8') as f:raw_ccf = json.load(f)
# 构建两个查找表：简称 → 等级，全称 → 等级
ccf_short_map = { item['short_name'].upper(): item['category'] for item in raw_ccf }
ccf_full_map  = { item['full_name'].upper():  item['category'] for item in raw_ccf }
with open('arxiv_process_outputs\\cas_mapping.json', 'r', encoding='utf-8') as f:
    cas_map = json.load(f)  # 中科院分区
with open('arxiv_process_outputs\\jcr_mapping.json', 'r', encoding='utf-8') as f:
    jcr_map = json.load(f)  # JCR 分区

# 动态获取 arXiv 分类代码到名称映射
def fetch_arxiv_category_map():
    url = 'https://arxiv.org/category_taxonomy'
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        category_map = {}
        for group in soup.find_all('div', class_='columns divided'):
            for item in group.find_all('div', class_='column is-one-fifth'):
                h4 = item.find('h4')
                if not h4:
                    continue
                code_node = h4.contents[0] if h4.contents else None
                code = str(code_node).strip() if code_node else None
                span = h4.find('span')
                if span and span.text:
                    name = span.text.strip()
                    if name.startswith('(') and name.endswith(')'):
                        name = name[1:-1].strip()
                else:
                    name = None
                if code:
                    category_map[code] = name or code
        return category_map
    except Exception as e:
        print(f"警告: 无法获取 arXiv 分类表，使用本地映射。错误: {e}")
        return {}

def load_category_map_cache():
    cache_file = 'arxiv_process_outputs\\arxiv_categories.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    category_map = fetch_arxiv_category_map()
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(category_map, f, ensure_ascii=False, indent=2)
    return category_map

CATEGORY_MAP = load_category_map_cache()

def normalize_str(s):
    if s is None:
        return None
    s = s.strip()
    s = html.unescape(s.strip())
    return s if s else None

def translate_categories(cat_list):
    if not cat_list:
        return None
    translated = []
    for code in cat_list:
        name = CATEGORY_MAP.get(code)
        translated.append(name if name else code)
    return ';'.join(translated)

def fetch_arxiv_papers(query, year_start, year_end, per_page, pages):
    client = arxiv.Client()
    results = []
    
    # 分批获取，避免一次性请求过多
    for page in range(pages):
        try:
            search = arxiv.Search(
                query=query,
                max_results=per_page,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending,
            )
            
            # 获取当前页的结果
            current_results = list(client.results(search))
            
            # 如果没有结果，说明已经到达末尾
            if not current_results:
                print(f"第 {page + 1} 页没有更多结果，停止获取")
                break
                
            # 筛选年份范围内的论文
            filtered_results = []
            for paper in current_results:
                if year_start <= paper.published.year <= year_end:
                    filtered_results.append(paper)
            
            results.extend(filtered_results)
            
            # 添加延时避免API限制
            time.sleep(1)
            
        except arxiv.UnexpectedEmptyPageError:
            print(f"第 {page + 1} 页返回空结果，可能已达到结果末尾")
            break
        except Exception as e:
            print(f"获取第 {page + 1} 页时发生错误: {e}")
            break
    
    return results

def get_openalex_info(doi=None, arxiv_id=None):
    if doi:
        api = f"https://api.openalex.org/works/doi:{doi}"
    elif arxiv_id:
        base_id = arxiv_id.split('v')[0]
        api = f"https://api.openalex.org/works/arxiv:{base_id}"
    else:
        return None, None
    try:
        r = requests.get(api, timeout=10)
        r.raise_for_status()
        data = r.json()
        venue = data.get('host_venue', {}).get('display_name')
        cites = data.get('cited_by_count')
        return venue, cites
    except Exception:
        return None, None

def get_crossref_info(doi):
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=10)
        r.raise_for_status()
        m = r.json().get('message', {})
        venue = m.get('container-title', [None])[0]
        cites = m.get('is-referenced-by-count')
        return venue, cites
    except Exception:
        return None, None

# ADS API 相关函数
dotenv.load_dotenv()
ADS_TOKEN = os.getenv('ADS_TOKEN')
if not ADS_TOKEN:
    raise RuntimeError("请先在环境变量 ADS_TOKEN 中设置您的 NASA ADS API token")
ADS_HEADERS = {"Authorization": f"Bearer {ADS_TOKEN}"}

# # 从 Google Scholar 获取期刊、平台、被引用数
# def fetch_from_scholar(arxiv_id, timeout=10):
#     base = arxiv_id.split('v')[0]
#     abs_url = f"https://arxiv.org/abs/{base}"
#     try:
#         resp = requests.get(abs_url, timeout=timeout)
#         resp.raise_for_status()
#         soup = BeautifulSoup(resp.text, 'html.parser')
#         extra = soup.find('div', class_='extra-ref-cite')
#         if extra:
#             ul = extra.find('ul')
#             if ul:
#                 li_list = ul.find_all('li')
#                 if len(li_list) >= 2:
#                     link = li_list[1].find('a')
#                     if link and link.get('href'):
#                         gs_url = link['href']
#                         gs_resp = requests.get(gs_url, timeout=timeout)
#                         gs_resp.raise_for_status()
#                         gs_soup = BeautifulSoup(gs_resp.text, 'html.parser')
#                         # 先检测是否存在 div.gs_r，若存在，说明未找到对应论文
#                         if gs_soup.find('div', class_='gs_r'):
#                             return None, None, None
#                         # 否则继续解析期刊、平台、被引用数
#                         info_div = gs_soup.find('div', class_='gs_a gs_fma_p')
#                         journal, platform = None, None
#                         if info_div and info_div.text:
#                             parts = [p.strip() for p in info_div.text.split('-')]
#                             if parts:
#                                 journal = parts[0].split(',')[0].strip()
#                             if len(parts) > 1:
#                                 platform = parts[1].strip()
#                         cite_div = gs_soup.find('div', class_='gs_fl gs_flb')
#                         cites = None
#                         if cite_div:
#                             a_tags = cite_div.find_all('a')
#                             if len(a_tags) >= 3:
#                                 text = a_tags[2].text
#                                 cites = text.split(':')[-1].strip()
#                         return normalize_str(journal or '未找到'), normalize_str(platform or '未找到'), (int(cites) if cites and cites.isdigit() else None)
#     except Exception as e:
#         print(f"Warning: unable to scrape arXiv/Scholar for {arxiv_id}: {e}")
#     return None, None, None

def get_ads_info(arxiv_id):
    # 去除版本号
    base_id = arxiv_id.split('v')[0]
    # 构造查询
    params = {
        'q': f'identifier:arXiv:{base_id}',
        'fl': 'keyword,pub,citation_count',
        'rows': 1
    }
    try:
        r = requests.get('https://api.adsabs.harvard.edu/v1/search/query', headers=ADS_HEADERS, params=params, timeout=10)
        r.raise_for_status()
        docs = r.json().get('response', {}).get('docs', [])
        if not docs:
            return None, None
        doc = docs[0]
        pub = doc.get('pub')
        cites = doc.get('citation_count')
        return  pub, cites
    except Exception as e:
        print(f"ADS 查询失败 for {arxiv_id}: {e}")
        return None, None

# 根据期刊名称，在两个映射里查等级。返回：多个用 ';' 分隔，找不到返回 '未找到'
def get_journal_level(venue):
    if not venue:
        return '未找到'
    key = venue.upper()
    levels = []

    # CAS
    if key in cas_map:
        levels.append(cas_map[key])
    # JCR
    if key in jcr_map:
        levels.append(jcr_map[key])
    # CCF —— 先试简称，再试全称
    if key in ccf_short_map:
        levels.append(ccf_short_map[key])
    elif key in ccf_full_map:
        levels.append(ccf_full_map[key])

    if not levels:
        return '未找到'
    # 去重、保持顺序
    return ';'.join(dict.fromkeys(levels))

def main():
    parser = argparse.ArgumentParser(description="Scrape arXiv metadata with citation sources.")
    parser.add_argument('--query', required=True)
    parser.add_argument('--year-start', type=int, default=2023)
    parser.add_argument('--year-end', type=int, default=2024)
    parser.add_argument('--pages', type=int, default=2)
    parser.add_argument('--per-page', type=int, default=100)
    parser.add_argument('--output', default='arxiv_process_outputs\\arxiv_full.csv')
    args = parser.parse_args()

    papers = fetch_arxiv_papers(
        args.query, args.year_start, args.year_end, args.per_page, args.pages
    )
    records = []
    for paper in tqdm(papers, desc="Processing papers", unit="paper"):
        aid = normalize_str(paper.entry_id.split('/')[-1])
        title = normalize_str(paper.title)
        authors = normalize_str(", ".join(a.name for a in paper.authors))
        abstract = normalize_str(paper.summary.replace('\n', ' '))
        doi = normalize_str(paper.doi)
        categories = translate_categories(paper.categories)

        pub, cites = get_ads_info(aid)
        if pub is None and (aid or doi):
            pub, cites = get_openalex_info(doi, aid)
        elif pub is None and doi:
            pub, cites = get_crossref_info(doi)
        pub = normalize_str(pub)
        journal_level = get_journal_level(pub)

        records.append({
            'arxiv_id': aid,
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'doi': doi,
            'categories': categories,
            'pub': pub,
            'citations': cites,
            'journal_level': journal_level,
        })
        time.sleep(3)

    df = pd.DataFrame(records)

    # 归一化所有字符串列
    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        df[col] = df[col].map(lambda x: normalize_str(x))

    # 将 'arXiv e-prints' 统一为 'arXiv'
    df['pub'] = df['pub'].replace('arXiv e-prints', 'arXiv')

    # 先保存完整表
    os.makedirs('arxiv_process_outputs', exist_ok=True)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"Saved full dataset ({len(records)} records) to {args.output}")

    # 基于 pub 分类并保存子集
    df_no_pub = df[df['pub'].isna() | (df['pub'] == '')]
    df_arxiv_only = df[df['pub'] == 'arXiv']
    df_published = df[~df.index.isin(df_no_pub.index) & ~df.index.isin(df_arxiv_only.index)]

    df_no_pub.to_csv('arxiv_process_outputs\\no_pub.csv', index=False, encoding='utf-8-sig')
    df_arxiv_only.to_csv('arxiv_process_outputs\\arxiv_only.csv', index=False, encoding='utf-8-sig')
    df_published.to_csv('arxiv_process_outputs\\published_journals.csv', index=False, encoding='utf-8-sig')

    print("Split and saved: no_pub.csv, arxiv_only.csv, published_journals.csv")

if __name__ == '__main__':
    main()