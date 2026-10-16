import io
p="src/IGWO.m"; s=io.open(p,encoding="utf-8").read()
old = """    for sIdx=sel
        if fe>=budget, break; end
        i=1+mod(sIdx-1,N);
        v=f(C(sIdx,:));
        fe=fe+1; bestEver=min(bestEver,v); T(fe)=bestEver;
        if v<pBestScore(i), pBest(i,:)=C(sIdx,:); pBestScore(i)=v; end
    end"""
new = """    if numel(sel)==2*N && isequal(sel,1:2*N)
        for i=1:N                       % original interleaved order (diagnostic)
            if fe>=budget, break; end
            vg=f(C(i,:)); fe=fe+1; bestEver=min(bestEver,vg); T(fe)=bestEver;
            if vg<pBestScore(i), pBest(i,:)=C(i,:); pBestScore(i)=vg; end
            if fe>=budget, break; end
            vd=f(C(N+i,:)); fe=fe+1; bestEver=min(bestEver,vd); T(fe)=bestEver;
            if vd<pBestScore(i), pBest(i,:)=C(N+i,:); pBestScore(i)=vd; end
        end
    else
    for sIdx=sel
        if fe>=budget, break; end
        i=1+mod(sIdx-1,N);
        v=f(C(sIdx,:));
        fe=fe+1; bestEver=min(bestEver,v); T(fe)=bestEver;
        if v<pBestScore(i), pBest(i,:)=C(sIdx,:); pBestScore(i)=v; end
    end
    end"""
assert old in s
io.open("_IGWO_diag.m","w",encoding="utf-8").write(s.replace(old,new))
print("diagnostic copy written")
