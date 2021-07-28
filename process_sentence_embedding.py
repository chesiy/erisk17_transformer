import os
import re
from sentence_transformers import SentenceTransformer
import xml.dom.minidom
import string
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch
from cluster_summary import get_kmeans_centroid_ids, get_cluster_summary
import numpy as np

# %%
sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# %%
def get_input_data(file, path, k):
    post_num = 0
    dom = xml.dom.minidom.parse(path + "/" + file)
    collection = dom.documentElement
    title = collection.getElementsByTagName('TITLE')
    text = collection.getElementsByTagName('TEXT')
    posts = []
    for i in range(len(title)):
        post = title[i].firstChild.data + ' ' + text[i].firstChild.data
        post = re.sub('\n', ' ', post)
        if len(post) > 0:
            posts.append(post.strip())
            post_num = post_num + 1
        if post_num == k:
            break
    # print(posts)
    # print(post_num)
    # print(file,'ok!')
    return posts, post_num


# %%
# with open("processed/miniLM_L6_embs.pkl", "wb") as f:
#     pickle.dump({
#         "train_posts": train_posts,
#         "train_mappings": train_mappings,
#         "train_labels": train_tags,
#         "train_embs": train_embs,
#         "test_posts": test_posts,
#         "test_mappings": test_mappings,
#         "test_labels": test_tags,
#         "test_embs": test_embs
#     }, f)

# with open("processed/miniLM_L6_embs.pkl", "rb") as f:
#     data = pickle.load(f)
#
# train_posts = data["train_posts"]
# train_mappings = data["train_mappings"]
# train_tags = data["train_labels"]
# train_embs = data["train_embs"]
# test_posts = data["test_posts"]
# test_mappings = data["test_mappings"]
# test_tags = data["test_labels"]
# test_embs = data["test_embs"]


def get_chunk_summary(chunkid, postnum_recorder, chunknum):
    test_posts = []
    test_tags = []
    test_mappings = []
    for base_path in ["negative_examples_test", "positive_examples_test"]:
        base_path = "dataset/" + base_path
        filenames = sorted(os.listdir(base_path))
        i = 0
        for fname in filenames:
            k = postnum_recorder[base_path][i] * (chunkid + 1) / chunknum
            posts, post_num = get_input_data(fname, base_path, k)
            test_mappings.append(list(range(len(test_posts), len(test_posts) + post_num)))
            test_posts.extend(posts)
            test_tags.append(int("positive" in base_path))
            i += 1

    test_embs = sbert.encode(test_posts, convert_to_tensor=False)

    test_posts = np.array(test_posts)
    test_mappings = np.array(test_mappings)
    test_tags = np.array(test_tags)

    # %%
    # for K in [8, 16, 32, 64]:
    for K in [16]:
        os.makedirs(f"./processed/kmeans{K}", exist_ok=True)
        os.makedirs(f"./processed/kmeans{K}/test", exist_ok=True)

        for id0, members in enumerate(tqdm(test_mappings, desc=f"K={K}, test")):
            user_posts1 = [test_posts[i] for i in members]
            user_embs1 = test_embs[members]
            label1 = test_tags[id0]
            summaries1 = get_cluster_summary(user_posts1, user_embs1, K=K)
            with open(f"./processed/kmeans{K}/test/{id0:06}_{label1}.txt", "w") as f:
                f.write("\n".join(summaries1))


def get_sample_summary(k):
    test_posts = []
    test_tags = []
    test_mappings = []
    for base_path in ["negative_examples_test", "positive_examples_test"]:
        base_path = "dataset/" + base_path
        filenames = sorted(os.listdir(base_path))
        i = 0
        for fname in filenames:
            posts, post_num = get_input_data(fname, base_path, k)
            test_mappings.append(list(range(len(test_posts), len(test_posts) + post_num)))
            test_posts.extend(posts)
            test_tags.append(int("positive" in base_path))
            i += 1

    test_embs = sbert.encode(test_posts, convert_to_tensor=False)

    test_posts = np.array(test_posts)
    test_mappings = np.array(test_mappings)
    test_tags = np.array(test_tags)

    # %%
    # for K in [8, 16, 32, 64]:
    for K in [16]:
        os.makedirs(f"./processed/kmeans{K}", exist_ok=True)
        os.makedirs(f"./processed/kmeans{K}/test", exist_ok=True)

        for id0, members in enumerate(tqdm(test_mappings, desc=f"K={K}, test")):
            user_posts1 = [test_posts[i] for i in members]
            user_embs1 = test_embs[members]
            label1 = test_tags[id0]
            summaries1 = get_cluster_summary(user_posts1, user_embs1, K=K)
            with open(f"./processed/kmeans{K}/test/{id0:06}_{label1}.txt", "w") as f:
                f.write("\n".join(summaries1))


def get_sample_summary_v2(file, path):
    test_posts = []
    test_tags = []
    test_mappings = []
    posts, post_num = get_input_data(file, path, -1)

    curposts = []
    for i in range(post_num):
        test_mappings.append(list(range(len(test_posts), len(test_posts) + i + 1)))
        curposts.append(posts[i])
        test_posts.extend(curposts)
        test_tags.append(int("positive" in path))

    test_embs = sbert.encode(test_posts, convert_to_tensor=False)

    test_posts = np.array(test_posts)
    test_mappings = np.array(test_mappings)
    test_tags = np.array(test_tags)



    # for K in [8, 16, 32, 64]:
    for K in [16]:
        # os.makedirs(f"./processed/kmeans{K}", exist_ok=True)
        files = os.listdir(f"./processed/kmeans{K}/test")
        for file in files:
            os.remove(f"./processed/kmeans{K}/test/{file}")

        os.makedirs(f"./processed/kmeans{K}/test", exist_ok=True)

        for id0, members in enumerate(tqdm(test_mappings, desc=f"K={K}, test")):
            user_posts1 = [test_posts[i] for i in members]
            user_embs1 = test_embs[members]
            label1 = test_tags[id0]
            summaries1 = get_cluster_summary(user_posts1, user_embs1, K=K)
            with open(f"./processed/kmeans{K}/test/{id0:06}_{label1}.txt", "w") as f:
                f.write("\n".join(summaries1))
