function [br, ep] = make_skel(x)
    br = branchedPoints(x);
    ep = endPoints(x);
    br = br - br.*ep;

end

function br = branchedPoints(skel)
    xbranch0 = [1 0 1;0 1 0; 1 0 1];
    xbranch1 = [0 1 0; 1 1 1; 0 1 0];
    tbranch0 = [0 0 0; 1 1 1; 0 1 0];
    tbranch1 = flipud(tbranch0);
    tbranch2 = tbranch0';
    tbranch3 = fliplr(tbranch2);

    tbranch4 = [1 0 1; 0 1 0; 1 0 0];
    tbranch5 = flipud(tbranch4);
    tbranch6 = fliplr(tbranch4);
    tbranch7 = fliplr(tbranch5);

    ybranch0 = [1 0 1; 0 1 0; 0 1 0];
    ybranch1 = flipud(ybranch0);
    ybranch2 = ybranch0';
    ybranch3 = fliplr(ybranch2);

    ybranch4 = [0 1 0; 1 1 0;0 0 1];

    ybranch5 = flipud(ybranch4);
    ybranch6 = fliplr(ybranch4);
    ybranch7 = fliplr(ybranch5);

    offbranch0 = [0 1 0; 1 1 0; 1 0 1];
    offbranch1 = flipud(offbranch0);
    offbranch2 = fliplr(offbranch0);
    offbranch3 = fliplr(offbranch1);
    offbranch4 = offbranch0';
    offbranch5 = flipud(offbranch4);
    offbranch6 = fliplr(offbranch4);
    offbranch7 = fliplr(offbranch5);

    clustbranch0 =[0 1 1; 0 1 1; 1 0 0];

    clustbranch1 = flipud(clustbranch0);
    clustbranch2 = fliplr(clustbranch0);
    clustbranch3 = fliplr(clustbranch1);

    clustbranch4 = [1 1 1; 0 1 1; 1 0 0];

    clustbranch5 = flipud(clustbranch4);
    clustbranch6 = fliplr(clustbranch4);
    clustbranch7 = fliplr(clustbranch5);

    clustbranch8 = [1 1 1; 0 1 1; 1 0 1];

    clustbranch9 = flipud(clustbranch8);
    clustbranch10 = fliplr(clustbranch8);
    clustbranch11 = fliplr(clustbranch9);

    crossbranch0 = [1 0 0; 1 1 1; 0 1 0];

    crossbranch1 = flipud(crossbranch0);
    crossbranch2 = fliplr(crossbranch0);
    crossbranch3 = fliplr(crossbranch1);
    crossbranch4 = crossbranch0';
    crossbranch5 = flipud(crossbranch4);
    crossbranch6 = fliplr(crossbranch4);
    crossbranch7 = fliplr(crossbranch5);

    br1 = bwhitmiss(skel, xbranch0);
    br2 = bwhitmiss(skel, xbranch1);
    br3 = bwhitmiss(skel, tbranch0);
    br4 = bwhitmiss(skel, tbranch1);
    br5 = bwhitmiss(skel, tbranch2);
    br6 = bwhitmiss(skel, tbranch3);
    br7 = bwhitmiss(skel, tbranch4);
    br8 = bwhitmiss(skel, tbranch5);
    br9 = bwhitmiss(skel, tbranch6);
    br10 = bwhitmiss(skel, tbranch7);
    br11 = bwhitmiss(skel, ybranch0);
    br12 = bwhitmiss(skel, ybranch1);
    br13 = bwhitmiss(skel, ybranch2);
    br14 = bwhitmiss(skel, ybranch3);
    br15 = bwhitmiss(skel, ybranch4);
    br16 = bwhitmiss(skel, ybranch5);
    br17 = bwhitmiss(skel, ybranch6);
    br18 = bwhitmiss(skel, ybranch7);
    br19 = bwhitmiss(skel, offbranch0);
    br20 = bwhitmiss(skel, offbranch1);
    br21 = bwhitmiss(skel, offbranch2);
    br22 = bwhitmiss(skel, offbranch3);
    br23 = bwhitmiss(skel, offbranch4);
    br24 = bwhitmiss(skel, offbranch5);
    br25 = bwhitmiss(skel, offbranch6);
    br26 = bwhitmiss(skel, offbranch7);
    br27 = bwhitmiss(skel, clustbranch0);
    br28 = bwhitmiss(skel, clustbranch1);
    br29 = bwhitmiss(skel, clustbranch2);
    br30 = bwhitmiss(skel, clustbranch3);
    br31 = bwhitmiss(skel, clustbranch4);
    br32 = bwhitmiss(skel, clustbranch5);
    br33 = bwhitmiss(skel, clustbranch6);
    br34 = bwhitmiss(skel, clustbranch7);
    br35 = bwhitmiss(skel, clustbranch8);
    br36 = bwhitmiss(skel, clustbranch9);
    br37 = bwhitmiss(skel, clustbranch10);
    br38 = bwhitmiss(skel, clustbranch11);
    br39 = bwhitmiss(skel, crossbranch0);
    br40 = bwhitmiss(skel, crossbranch1);
    br41 = bwhitmiss(skel, crossbranch2);
    br42 = bwhitmiss(skel, crossbranch3);
    br43 = bwhitmiss(skel, crossbranch4);
    br44 = bwhitmiss(skel, crossbranch5);
    br45 = bwhitmiss(skel, crossbranch6);
    br46 = bwhitmiss(skel, crossbranch7);

    br = br1+br2+br3+br4+br5+br6+br7+br8+br9+br10+br11+br12+br13+br14+br15+br16+br17+br18+br19+br20+br21+br22+br23+br24 ...
         +br25+br26+br27+br28+br29+br30+br31+br32+br33+br34+br35+br36+br37+br38+br39+br40+br41+br42+br43+br44+br45+br46;
    br = 1*(br>0);
end

function ep = endPoints(skel)
    endpoint1 = [0 0 0; 0 1 0; 0 1 0];

    endpoint2 = [ 0 0 0; 0 1 0; 0 0 1];

    endpoint3 = [0 0 0; 0 1 1; 0 0 0];

    endpoint4 = [0 0 1; 0 1 0 ; 0 0 0];

    endpoint5 = [0 1 0; 0 1 0 ; 0 0 0];
    endpoint6 = [1 0 0; 0 1 0 ; 0 0 0];

    endpoint7 = [0 0 0; 1 1 0 ; 0 0 0];

    endpoint8 =[0 0 0; 0 1 0 ; 1 0 0];

    endpoint9 = [0 0 0; 0 1 0 ; 0 0 0];

    endpoint10 = [1 0 0; 1 1 0 ; 0 0 0];
    endpoint11 = [1 1 0; 0 1 0 ; 0 0 0];
    endpoint12 = [0 1 1; 0 1 0 ; 0 0 0];
    endpoint13 = [0 0 1; 0 1 1 ; 0 0 0];
    endpoint14 = [0 0 0; 0 1 1 ; 0 0 1];
    endpoint15 = [0 0 0; 0 1 0 ; 0 1 1];
    endpoint16 = [0 0 0; 0 1 0 ; 1 1 0];
    endpoint17 = [0 0 0; 1 1 0 ; 1 0 0];

    ep1 = bwhitmiss(skel, endpoint1, 1 - endpoint1);
    ep2 = bwhitmiss(skel, endpoint2, 1- endpoint2);
    ep3 = bwhitmiss(skel, endpoint3, 1- endpoint3);
    ep4 = bwhitmiss(skel, endpoint4, 1-endpoint4);
    ep5 = bwhitmiss(skel, endpoint5, 1- endpoint5);
    ep6 = bwhitmiss(skel, endpoint6, 1-endpoint6);
    ep7 = bwhitmiss(skel, endpoint7, 1-endpoint7);
    ep8 = bwhitmiss(skel, endpoint8, 1-endpoint8);
    ep9 = bwhitmiss(skel, endpoint9,1-endpoint9);
    ep10 = bwhitmiss(skel, endpoint10, 1-endpoint10);
    ep11 = bwhitmiss(skel, endpoint11, 1- endpoint11);
    ep12 = bwhitmiss(skel, endpoint12, 1-endpoint12);
    ep13 = bwhitmiss(skel, endpoint13, 1-endpoint13);
    ep14 = bwhitmiss(skel, endpoint14, 1-endpoint14);
    ep15 = bwhitmiss(skel, endpoint15,1-endpoint15);
    ep16 = bwhitmiss(skel, endpoint16,1-endpoint16);
    ep17 = bwhitmiss(skel, endpoint17,1-endpoint17);

    ep = ep1 + ep2 + ep3 + ep4 + ep5 + ep6 + ep7 + ep8 + ep9 ...
        + ep10 + ep11 + ep12 + ep13 + ep14 +ep15 + ep16 + ep17;
    ep = 1*(ep>0);
end
