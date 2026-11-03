"""Host-only checks. HIP correctness and speed still require the actual GPU."""
import ast
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CU = ROOT / 'exllamav3/exllamav3_ext/quant/exl3_dec.cu'


def test_actual_tile_format_and_windows(tmp_path):
    source = CU.read_text()
    fmt = re.search(r'template <int KB2>\nstruct Fmt\n\{.*?\n\};', source, re.S).group()
    window = re.search(r'template <int KB2, int P, bool MASK>\n.*?\n\}', source, re.S).group()
    preamble = '''#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#define __device__
#define __forceinline__ inline
static uint32_t __builtin_amdgcn_ubfe(uint32_t x,int o,int w) {return (x>>o)&((1u<<w)-1);}
static uint32_t __builtin_amdgcn_alignbit(uint32_t h,uint32_t l,int s) {return uint32_t(((uint64_t(h)<<32)|l)>>s);}
#ifndef PORT_UBFE16  // newer exl3_dec.cu goes through the port macros
#define PORT_UBFE16(w, s) __builtin_amdgcn_ubfe(w, s, 16)
#define PORT_ALIGNBIT(hi, lo, s) __builtin_amdgcn_alignbit(hi, lo, s)
#endif
'''
    body = r'''
template<int B, size_t... P> bool check(const uint32_t* w,std::index_sequence<P...>) {
    const uint32_t got[] = {window<B,int(P),true>(w)...};
    const uint32_t raw[] = {window<B,int(P),false>(w)...};
    for(int p=0;p<256;p++) {
        int end=(p+257)*(B/2), start=end-16;
        uint32_t a=w[(start/32)%Fmt<B>::WORDS], b=w[((end-1)/32)%Fmt<B>::WORDS];
        uint32_t want=uint32_t(((uint64_t(a)<<32)|b)>>(((end-1)/32+1)*32-end))&65535;
        if(got[p]!=want || (raw[p]&65535)!=want) return false;
    }
    return true;
}
template<int B> bool run() {
    uint32_t w[Fmt<B>::WORDS]; srand(11+B);
    for(int t=0;t<1000;t++) {
        for(auto &v:w) v=(uint32_t(rand())<<16)^uint32_t(rand());
        if(!check<B>(w,std::make_index_sequence<256>{})) return false;
    }
    return true;
}
int main() {return !(run<10>() && run<12>() && run<14>() && run<16>());}
'''
    cpp = tmp_path / 'check.cpp'
    cpp.write_text(preamble + fmt + window + body)
    exe = tmp_path / 'check'
    subprocess.run(['g++','-std=c++17','-O1','-fsanitize=address,undefined',
                    '-fno-sanitize-recover=all',str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0'})


def helpers():
    tree = ast.parse((ROOT / 'exllamav3/modules/quant/exl3.py').read_text())
    nodes = [n for n in tree.body if
             isinstance(n, ast.FunctionDef) and n.name in ('dec_supported','dec_moe_supported') or
             isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id in
                ('_DEC_KB2','_DEC_KB2_DENSE','_DENSE_K8_SHAPES') for t in n.targets)]
    ns = {'os':os,'torch':NS(version=NS(hip=True)), 'EXL3_DEC':True,
          'ext':NS(exl3_dec_gemv=True,exl3_dec_moe=True)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'eligibility','exec'),ns)
    return ns


def inner(k=8, shape=(2560,10240)):
    i,o=shape
    return NS(K=k,mul1=True,mcg=False,in_features=i,out_features=o,
              trellis=NS(shape=(i//16,o//16,128)),suh=1,svh=1)


class DenseK8Checks(unittest.TestCase):
    def test_tile_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_actual_tile_format_and_windows(Path(tmp))

    def test_flag_shapes_and_moe_isolation(self):
        for flag in [None, '0', '1', 'true', '01']:
            with self.subTest(flag=flag), patch.dict(os.environ):
                os.environ.pop('EXL3_DENSE_K8', None)
                if flag is not None:
                    os.environ['EXL3_DENSE_K8'] = flag
                ns = helpers()
                for shape in ns['_DENSE_K8_SHAPES']:
                    self.assertEqual(ns['dec_supported'](inner(shape=shape)), flag == '1')
                    self.assertFalse(ns['dec_moe_supported'](inner(shape=shape)))
                for shape in [(640,2560),(2560,640),(512,512)]:
                    self.assertFalse(ns['dec_supported'](inner(shape=shape)))
                for k in [2,2.5,3,4,5,6,7]:
                    self.assertTrue(ns['dec_supported'](inner(k=k)))
                for k in [7.5,8.5,9]:
                    self.assertFalse(ns['dec_supported'](inner(k=k)))

    def test_rejects_unsupported_weights(self):
        with patch.dict(os.environ, {'EXL3_DENSE_K8':'1'}):
            for attribute,value in [('mul1',False),('mcg',True),('suh',None),('svh',None)]:
                x = inner()
                setattr(x,attribute,value)
                self.assertFalse(helpers()['dec_supported'](x))

if __name__ == '__main__':
    unittest.main()
