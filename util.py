import os
from io import StringIO
from multiprocessing import Process, cpu_count
import joblib


def get_cpu_count():
    return cpu_count()


def dump(fn, data):
    # pkl_file = open(fn, 'w')
    # pickle.dump(data, pkl_file)
    # pkl_file.close()

    # sklearn.externals joblib is faster
    joblib.dump(data, fn)


def load(fn):
    # pkl_file = open(fn, 'r')
    # res = pickle.load(pkl_file)
    # pkl_file.close()

    res = joblib.load(fn)
    return res


# ----------- multiprocessing ---------------------- #

def single_process(func, index, para_l, prefix, args):
    results = []
    if para_l is not None:
        for p in para_l:
            #print 'single_process', index, p,
            res = func(p, **args)
            #print re
            results.append(res)
    else:
        results = func(**args)
    fn = 'tmp/%s_%d.p' % (prefix, index)
    dump(fn, results)


def multiprocess(func, paras=[], name=None, n_processes=1, **args):

    result = []
    if n_processes == 1:
        if paras == []:
            return func(**args)
        else:
            for p in paras:
                result.append(func(p, **args))
        return result
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    l = len(paras)
    size = l // n_processes
    j = 0
    jobs = []
    prefix = func.__name__ + '_%s' % os.getpid()
    for i in range(n_processes):
        # print i, self.feature.para_range[j:end_]
        if i < l % n_processes:
            gap = size + 1
        else:
            gap = size
        if l == 0:
            tmp_paras = None
        else:
            tmp_paras = paras[j:j + gap]
        if name == None:
            namep = 'process%d' % (i+1)
        else:
            namep = name[i]
        p = Process(target=single_process, name=namep, args=(func, i, tmp_paras, prefix, args))
        jobs.append(p)
        p.start()
        j += gap
        if (j >= l) and (l != 0):
            break
    n_processes = len(jobs)
    for i in range(n_processes):
        jobs[i].join()
    return collect_data(prefix, n_processes)


def collect_data(prefix, num):
    data = []
    for i in range(num):
        fn = 'tmp/%s_%d.p' % (prefix, i)
        d = load(fn)
        data.extend(d)
        os.remove(fn)
    return data


def process_uploaded_paper_data(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()
    return string_data






