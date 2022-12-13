from test_pipeline import chart_process

images = ["/home/blee/nfs/challenge_2022/DeepRule_tmp/imgs/796930_figure_2.png", "/home/blee/nfs/challenge_2022/DeepRule_tmp/imgs/1111_bar_AGC.png", "/home/blee/nfs/challenge_2022/DeepRule_tmp/pie_ex.png", "/home/blee/nfs/challenge_2022/DeepRule_tmp/imgs/160124_figure_1.png"]
for img in images:
    result = chart_process(img)
    print(result)
    import pdb; pdb.set_trace()