/*
 * 美yond 写真ページ: 顔の位置合わせと、アフターの明るさ（お客様ページ index.html と管理画面 admin/index.html の両方で使う）
 *
 * 顔の位置: 管理画面で写真を入れるとき、スタッフのスマホの中で眉・目・鼻・口の位置（20か所）を読み取り、
 *   その写真のドライブの「説明」に数字だけ書いておく。写真そのものはどこにも送らない。
 *   読み取りは Google の MediaPipe（顔の点を478か所見つける仕組み）。管理画面だけが読み込む（初回だけ約15MB）
 *   ベッドに寝たお客様を頭の側から撮ると顔が上下さかさまに写る。点の読み取りはまっすぐの顔しか正しく読めないので、
 *   先に顔の向き（0・90・180・270度）を顔の見つけやすさで決め、その向きに回してから読む
 * 位置合わせ: お客様ページでは、その数字から「拡大・回転・移動」だけで、どの写真も同じ場所・同じ大きさに顔が来るよう切り抜く。
 *   顔の形は変えない（縦横の伸び縮みやゆがみは入れない）。元の写真はドライブにそのまま残る
 * 明るさ: iPhone の写真アプリの「明るさ」と同じ曲線。Mac の Core Image（写真アプリと同じ仕組み）で、
 *   目盛り -100〜100 を 10 きざみで測った表を持ち、間はなめらかにつなぐ（ずれは 256 段階で1段まで）
 */
(function () {
'use strict';

const LIB = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const DETECTOR = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite';

// 読み取る点（MediaPipe の番号）と、合わせるときの重み。ずれにくい目と鼻を重く、表情で動く眉・口・あごは軽く
const POINTS = [
  [70, 1], [105, 1], [107, 1], [300, 1], [334, 1], [336, 1], // 眉（外・中・内）右、左
  [33, 3], [133, 3], [263, 3], [362, 3],                     // 目じり・目がしら 右、左
  [168, 2], [1, 2], [2, 2], [98, 2], [327, 2],               // 鼻（つけ根・先・下・小鼻の右・左）
  [61, 1], [291, 1], [0, 1], [17, 1],                        // 口（口角の右・左・上くちびる・下くちびる）
  [152, 0.5],                                                // あご
];
// 正面の顔の標準の配置（両目の真ん中が 0,0、両目のあいだが 1。下が +）。見本の顔から左右をならして作った
const TEMPLATE = [-0.941,-0.249,-0.687,-0.351,-0.244,-0.263,0.941,-0.249,0.687,-0.351,0.244,-0.263,-0.737,-0.034,-0.263,0.034,0.737,-0.034,0.263,0.034,0.000,-0.019,0.000,0.689,0.000,0.769,-0.263,0.709,0.263,0.709,-0.458,0.988,0.458,0.988,0.000,0.967,0.000,1.211,0.000,1.709];
// 切り抜きの枠は 横3:縦4。両目の真ん中を横の真ん中・上から42%に置き、両目のあいだを枠の幅の30%にする
// upright: false は撮ったときの向きのまま（上下さかさまに撮った写真は、さかさまのまま顔の位置だけそろえる）。true にすると顔をまっすぐ立てて見せる
const FRAME = { ratio: 4 / 3, cx: 0.5, cy: 0.42 * 4 / 3, iod: 0.30, upright: false };
const FILL = '#E4EDEC'; // 元の写真の外にはみ出たところの色（ページの写真枠と同じ）

/* ---------- 横顔（2026-11-17 から） ----------
 * 横顔は眉・目・鼻・口の点の読み取り（正面向けの仕組み）では顔が見つからない。そこで、壁と顔の境目に出る
 * 「横顔の線」（額 → 鼻のつけ根 → 鼻先 → 鼻の下 → 上くちびる）を読み、標準の横顔の線に拡大・回転・移動だけで重ねる。
 * 写真ごとに、その重ね方を表す4つの点（標準の形の上の正方形の角を写真の上に戻したもの）を説明に書く（face:v3）
 * 標準の横顔の線: 立った人が右を向いた形。鼻のつけ根が (0,0)、鼻の下が (0,1) あたり。実際の施術写真16枚の平均 */
const SIDE_CURVE = [-0.217,-2.303,-0.202,-2.245,-0.186,-2.187,-0.171,-2.128,-0.156,-2.070,-0.143,-2.011,-0.146,-1.951,-0.138,-1.910,-0.086,-1.940,-0.065,-1.933,-0.085,-1.876,-0.089,-1.820,-0.045,-1.789,-0.033,-1.730,-0.015,-1.672,0.009,-1.617,0.028,-1.560,0.051,-1.505,0.071,-1.448,0.082,-1.389,0.099,-1.332,0.109,-1.273,0.125,-1.215,0.138,-1.156,0.149,-1.097,0.158,-1.037,0.161,-0.979,0.172,-0.922,0.183,-0.866,0.179,-0.806,0.193,-0.748,0.196,-0.688,0.201,-0.628,0.204,-0.568,0.170,-0.537,0.144,-0.529,0.185,-0.485,0.143,-0.464,0.128,-0.408,0.100,-0.355,0.094,-0.310,0.050,-0.301,0.079,-0.248,0.071,-0.213,0.031,-0.195,0.049,-0.139,0.036,-0.083,0.050,-0.043,0.036,-0.014,0.058,0.035,0.064,0.093,0.094,0.129,0.110,0.186,0.137,0.240,0.148,0.291,0.184,0.318,0.204,0.375,0.234,0.427,0.261,0.480,0.294,0.529,0.319,0.583,0.341,0.639,0.350,0.699,0.346,0.756,0.326,0.813,0.290,0.861,0.246,0.901,0.192,0.928,0.134,0.945,0.075,0.957,0.022,0.984,-0.014,1.031,-0.048,1.072,0.007,1.094,0.020,1.140,-0.012,1.191,0.019,1.206,0.079,1.202,0.087,1.255,0.070,1.312,0.036,1.362,0.007,1.415,0.007,1.473,0.036,1.526,0.056,1.582,0.057,1.642,0.038,1.698,0.003,1.747,-0.030,1.798,-0.050,1.855,-0.060,1.914,-0.065,1.974,-0.063,2.034,-0.069,2.094,-0.075,2.154,-0.081,2.214];
const SIDE_KEYS = [-1, -1, 1, -1, 1, 1, -1, 1]; // 標準の形の上の4つの点（重ね方を写真に残すためだけの点）
const SIDE_LO = 0.9, SIDE_HI = 0.22; // 使う線の長さ: 鼻先から額の側へ・あごの側へ（顔の大きさに対する割合）
// 横顔の枠: 鼻のつけ根と鼻の下のあいだを枠の幅の16%にし、ほおのあたり（鼻のつけ根から後ろへ1.3・下へ0.6）を枠の真ん中に置く
const SIDE_FRAME = { unit: 0.16, ax: -1.3, ay: 0.6 };
const SEGMENTER = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite';

/* ---------- 説明に書く文字（位置は写真の幅・高さに対する割合） ----------
 *   正面: face:v2;a=横÷縦;p=x,y,x,y…（20か所）
 *   横顔: face:v3;a=横÷縦;m=0か1（左向きの顔なら1）;p=x,y…（4つの点）
 *   顔なし: face:none2（正面も横顔も見つからなかった）
 * v1 は向きを決めずに読んでいたので、上下さかさまの顔を読み違えていた → 使わずに読み直す
 * face:none（2026-11-17 より前の「見つからない」）は横顔を読んでいないので、もう一度読み取る */
function parse(s) {
  if (typeof s !== 'string' || s.indexOf('face:') !== 0) return null;
  if (s === 'face:none2') return 'none';
  const parts = s.slice(5).split(';');
  const side = parts[0] === 'v3';
  if (parts[0] !== 'v2' && !side) return null; // 書き方が古い・新しい → もう一度読み取る
  const kv = {};
  parts.slice(1).forEach(t => { const i = t.indexOf('='); if (i > 0) kv[t.slice(0, i)] = t.slice(i + 1); });
  const p = String(kv.p || '').split(',').map(Number);
  const aspect = Number(kv.a);
  if (p.length !== (side ? SIDE_KEYS.length : POINTS.length * 2) || p.some(v => !isFinite(v)) || !(aspect > 0)) return null;
  return side ? { side: true, mirror: kv.m === '1', aspect, p } : { aspect, p };
}
function format(aspect, p) {
  return 'face:v2;a=' + aspect.toFixed(4) + ';p=' + p.map(v => v.toFixed(4)).join(',');
}
function formatSide(aspect, mirror, p) {
  return 'face:v3;a=' + aspect.toFixed(4) + ';m=' + (mirror ? 1 : 0) + ';p=' + p.map(v => v.toFixed(4)).join(',');
}

/* ---------- 合わせ方: 写真の点を標準の配置に、拡大・回転・移動だけで重ねる（重みつきの最小二乗） ---------- */
function fit(m, w, h) {
  const n = m.side ? SIDE_KEYS.length / 2 : POINTS.length;
  const T = m.side ? SIDE_KEYS : TEMPLATE;
  const pts = [], wts = [];
  for (let i = 0; i < n; i++) { pts.push([m.p[i * 2] * w, m.p[i * 2 + 1] * h]); wts.push(m.side ? 1 : POINTS[i][1]); }
  const tp = [];
  for (let i = 0; i < n; i++) tp.push([T[i * 2], T[i * 2 + 1]]);
  const s = simFit(pts, tp, wts);
  return { ar: s.a, ai: s.b, bx: s.tx, by: s.ty };
}

/* 点の組 X を点の組 Y に、拡大・回転・移動だけで重ねる（重みつきの最小二乗）。Y ≒ [a -b; b a]·X + t */
function simFit(X, Y, W) {
  let sw = 0, mx = 0, my = 0, nx = 0, ny = 0;
  X.forEach((p, i) => { const w = W ? W[i] : 1; sw += w; mx += w * p[0]; my += w * p[1]; nx += w * Y[i][0]; ny += w * Y[i][1]; });
  mx /= sw; my /= sw; nx /= sw; ny /= sw;
  let a = 0, b = 0, d = 0;
  X.forEach((p, i) => {
    const w = W ? W[i] : 1, x = p[0] - mx, y = p[1] - my, u = Y[i][0] - nx, v = Y[i][1] - ny;
    a += w * (x * u + y * v); b += w * (x * v - y * u); d += w * (x * x + y * y);
  });
  a /= d; b /= d;
  return { a, b, tx: nx - (a * mx - b * my), ty: ny - (b * mx + a * my) };
}

// 説明の数字がこの写真のものとして使えるか（縦横の比が合わないときは、向きが変わったとみなして使わない）
function usable(m, w, h) {
  return !!(m && m !== 'none' && m.p && w && h && Math.abs(m.aspect - w / h) < 0.03);
}

/* ---------- 明るさ（iPhone の「明るさ」の目盛り -100〜100）の表 ---------- */
const TONE_B64 = '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wMHCw8PExcbHyMnKy8zMzc7P0NHS0tPU1dbX19jZ2trb3N3e3t/g4OHi4+Pk5eXm5+fo6enq6uvs7O3t7u7v8PDx8fLy8/P09PT19fb29/f3+Pj5+fn6+vr7+/v7/Pz8/f39/f7+/v7+////ABUZHB8hIyUmKCkqKy0uLzAyMzQ1Njg5Ojs9Pj9AQUNERUZHSUpLTE5PUFFSVFVWV1haW1xdXmBhYmNkZmdoaWpsbW5vcHFzdHV2d3l6e3x9foCBgoOEhYaIiYqLjI2OkJGSk5SVlpeZmpucnZ6foKGio6Smp6ipqqusra6vsLGys7S1tre4ubq7vL2+v8DBwsLDxMXGx8jJysvMzM3Oz9DR0dLT1NXW1tfY2dna29zc3d7f3+Dh4eLj4+Tl5ebn5+jp6erq6+vs7e3u7u/v8PDx8fLy8/P09PT19fb29vf39/j4+Pn5+fr6+vv7+/v8/Pz8/f39/f3+/v7+/v///wAeIyYoKiwuLzEyMzQ1Nzg5Ojs8Pj9AQUJDRUZHSElKTE1OT1BRU1RVVldYWltcXV5fYWJjZGVnaGlqa2xub3BxcnN1dnd4eXp8fX5/gIGDhIWGh4iKi4yNjo+QkZOUlZaXmJmam52en6ChoqOkpaanqKmrrK2ur7CxsrO0tba3uLm6u7y9vr/AwMHCw8TFxsfIycrKy8zNzs/Q0NHS09TV1dbX2NjZ2tvb3N3e3t/g4OHi4uPk5OXm5ufn6Onp6urr6+zt7e7u7+/w8PHx8fLy8/P09PT19fb29vf39/j4+Pn5+fn6+vr6+/v7+/z8/Pz9/f39/f3+/v7+/v////8AJywvMTM1Njg5Ojs8PT4/QUJDREVGR0hJS0xNTk9QUVJTVVZXWFlaW11eX2BhYmNlZmdoaWprbW5vcHFydHV2d3h5e3x9fn+AgoOEhYaHiIqLjI2Oj5CSk5SVlpeYmZqcnZ6foKGio6Slpqipqqusra6vsLGys7S1tre4ubq7vL2+v8DBwsPExMXGx8jJysvMzM3Oz9DR0tLT1NXW1tfY2dna29zc3d7e3+Dh4eLi4+Tk5ebm5+fo6enq6uvr7Ozt7e7u7+/w8PHx8vLy8/P09PT19fX29vb39/f4+Pj5+fn5+vr6+vv7+/v7/Pz8/Pz9/f39/f3+/v7+/v7/////';
let toneTable = null;
const lutCache = {};
function toneLUT(value) {
  const v = Math.max(-100, Math.min(100, Number(value) || 0));
  if (lutCache[v]) return lutCache[v];
  if (!toneTable) {
    const bin = atob(TONE_B64);
    toneTable = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) toneTable[i] = bin.charCodeAt(i);
  }
  const pos = (v + 100) / 10, lo = Math.min(19, Math.floor(pos)), t = pos - lo;
  const out = new Uint8Array(256);
  for (let i = 0; i < 256; i++) out[i] = Math.round(toneTable[lo * 256 + i] * (1 - t) + toneTable[(lo + 1) * 256 + i] * t);
  return (lutCache[v] = out);
}
function tone(ctx, w, h, value) {
  const lut = toneLUT(value);
  const data = ctx.getImageData(0, 0, w, h);
  const d = data.data;
  for (let i = 0; i < d.length; i += 4) { d[i] = lut[d[i]]; d[i + 1] = lut[d[i + 1]]; d[i + 2] = lut[d[i + 2]]; }
  ctx.putImageData(data, 0, 0);
}

// 枠（W×H）の四隅が写真の中に入るまで、枠の真ん中を軸に何倍にすればよいか（1 なら今のままで届いている）
function coverScale(M, w, h, W, H) {
  const inv = M.inverse();
  const inside = s => [[0, 0], [W, 0], [0, H], [W, H]].every(([x, y]) => {
    const p = inv.transformPoint(new DOMPoint(W / 2 + (x - W / 2) / s, H / 2 + (y - H / 2) / s));
    return p.x >= 0 && p.y >= 0 && p.x <= w && p.y <= h;
  });
  if (inside(1)) return 1;
  let lo = 1, hi = 2;
  while (!inside(hi) && hi < 64) hi *= 2;
  for (let i = 0; i < 20; i++) { const mid = (lo + hi) / 2; if (inside(mid)) hi = mid; else lo = mid; }
  return hi;
}
// 写真の外側を、写真の端の色をなめらかにのばしたもので埋める（鏡写しだと耳や顔が二重に見えるので使わない）。
// 写真をとても小さく（短い辺が6マス）してから引きのばすので、細かい模様は出ず、色だけがぼんやり続く
function fillEdges(ctx, img, w, h, M) {
  const n = 6, sw = w >= h ? Math.round(n * w / h) : n, sh = w >= h ? n : Math.round(n * h / w);
  const mid = document.createElement('canvas'); // 一度に小さくすると色がざらつくので、二段で小さくする
  mid.width = sw * 8; mid.height = sh * 8;
  mid.getContext('2d').drawImage(img, 0, 0, mid.width, mid.height);
  const sm = document.createElement('canvas');
  sm.width = sw; sm.height = sh;
  const sc = sm.getContext('2d');
  sc.imageSmoothingQuality = 'high';
  sc.drawImage(mid, 0, 0, sw, sh);
  const E = Math.max(w, h) * 4, cw = w / sw, ch = h / sh;
  ctx.setTransform(M);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  const d = (sx, sy, sW, sH, dx, dy, dW, dH) => ctx.drawImage(sm, sx, sy, sW, sH, dx, dy, dW, dH);
  d(0, 0, sw, sh, 0, 0, w, h);                         // 写真の下にも（ふちを透かしたところから下の色が見えないように）
  d(0, 0, 0.5, sh, -E, -ch / 2, E, h + ch);            // 左
  d(sw - 0.5, 0, 0.5, sh, w, -ch / 2, E, h + ch);      // 右
  d(0, 0, sw, 0.5, -cw / 2, -E, w + cw, E);            // 上
  d(0, sh - 0.5, sw, 0.5, -cw / 2, h, w + cw, E);      // 下
  d(0, 0, 0.5, 0.5, -E, -E, E, E);                     // 四隅
  d(sw - 0.5, 0, 0.5, 0.5, w, -E, E, E);
  d(0, sh - 0.5, 0.5, 0.5, -E, h, E, E);
  d(sw - 0.5, sh - 0.5, 0.5, 0.5, w, h, E, E);
  mid.width = mid.height = sm.width = sm.height = 0;
}
// 写真のふちを少しだけ透かす（のばした色とのさかい目に線が出ないように）。ふちの幅は短い辺の 2.5%
function feathered(img, w, h) {
  const s = Math.min(1, 2400 / Math.max(w, h));
  const cw = Math.max(1, Math.round(w * s)), chh = Math.max(1, Math.round(h * s));
  const c = document.createElement('canvas');
  c.width = cw; c.height = chh;
  const x = c.getContext('2d');
  x.imageSmoothingQuality = 'high';
  x.drawImage(img, 0, 0, cw, chh);
  const e = Math.max(2, Math.round(Math.min(cw, chh) * 0.025));
  x.globalCompositeOperation = 'destination-out';
  const band = (x0, y0, x1, y1, rx, ry, rw, rh) => {
    const g = x.createLinearGradient(x0, y0, x1, y1);
    g.addColorStop(0, 'rgba(0,0,0,1)'); g.addColorStop(1, 'rgba(0,0,0,0)');
    x.fillStyle = g; x.fillRect(rx, ry, rw, rh);
  };
  band(0, 0, e, 0, 0, 0, e, chh);
  band(cw, 0, cw - e, 0, cw - e, 0, e, chh);
  band(0, 0, 0, e, 0, 0, cw, e);
  band(0, chh, 0, chh - e, 0, chh - e, cw, e);
  return c;
}

/* ---------- 画面に出す絵を作る ----------
 * marks があれば顔の位置で切り抜いた 横3:縦4 の絵、なければ元の写真のまま（長い辺 1600 まで）。bright は明るさの目盛り（0 なら何もしない）
 * rot は管理画面で直した向き（元の写真を右に回す角度 0・90・180・270。直していない写真は null）。元の写真は回さず、見せるときだけ回す */
function render(img, opts) {
  opts = opts || {};
  const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
  const canvas = opts.canvas || document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const rot = ROTS.indexOf(Number(opts.rot)) >= 0 ? Number(opts.rot) : null;
  const aligned = usable(opts.marks, w, h);
  if (aligned) {
    const m = opts.marks, side = !!m.side;
    const f = fit(m, w, h);
    const turn = faceTurn(f, rot, side);
    // 標準の形の 1 が、枠の幅のどれだけか（正面: 両目のあいだ。横顔: 鼻のつけ根から鼻の下まで）
    const unit = side ? SIDE_FRAME.unit : FRAME.iod;
    // 枠の幅が元の写真の何ピクセル分か。それより細かくしても見た目は変わらないので、絵の大きさはそこまで（480〜1080）
    const crop = 1 / (unit * Math.hypot(f.ar, f.ai));
    const W = Math.round(Math.min(1080, Math.max(480, crop))), H = Math.round(W * FRAME.ratio);
    canvas.width = W;
    canvas.height = H;
    ctx.fillStyle = FILL;
    ctx.fillRect(0, 0, W, H);
    // まっすぐの顔を枠に置いてから、枠の真ん中を軸に turn 度まわす（枠は縦長のまま。180度なら目の高さは下から42%）
    const k = W * unit;
    let M = new DOMMatrix().translate(W / 2, H / 2).rotate(turn).translate(-W / 2, -H / 2);
    if (side) {
      // 横顔: ほおのあたりを枠の真ん中に（左向きの顔は、標準の形が左右反対なので、後ろ＝右になる）
      const ax = SIDE_FRAME.ax * (m.mirror ? -1 : 1), ay = SIDE_FRAME.ay;
      M = M.multiply(new DOMMatrix([k * f.ar, k * f.ai, -k * f.ai, k * f.ar, k * (f.bx - ax) + W / 2, k * (f.by - ay) + H / 2]));
    } else {
      M = M.multiply(new DOMMatrix([k * f.ar, k * f.ai, -k * f.ai, k * f.ar, k * f.bx + W * FRAME.cx, k * f.by + W * FRAME.cy]));
    }
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    // 写真が枠の端まで届かないとき（顔の近くで撮った・傾いている）は、写真の端の色をのばして余白を埋め、
    // 写真のふちを少し透かしてなじませる。顔の大きさと位置は変えない（ビフォーとアフターの大きさがそろったまま）
    const short = coverScale(M, w, h, W, H) > 1.001;
    if (short) fillEdges(ctx, img, w, h, M);
    ctx.setTransform(M);
    if (short) {
      const fc = feathered(img, w, h);
      ctx.drawImage(fc, 0, 0, w, h);
      fc.width = fc.height = 0;
    } else {
      ctx.drawImage(img, 0, 0, w, h);
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  } else {
    const s = Math.min(1, 1600 / Math.max(w, h));
    const dw = Math.max(1, Math.round(w * s)), dh = Math.max(1, Math.round(h * s));
    const r = rot || 0;
    canvas.width = r % 180 ? dh : dw;
    canvas.height = r % 180 ? dw : dh;
    ctx.imageSmoothingQuality = 'high';
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.rotate(r * Math.PI / 180);
    ctx.drawImage(img, -dw / 2, -dh / 2, dw, dh);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }
  if (opts.bright) {
    try { tone(ctx, canvas.width, canvas.height, opts.bright); } catch (e) { /* 読めない絵のときは明るさだけあきらめる */ }
  }
  return { canvas, aligned };
}

const ROTS = [0, 90, 180, 270];
/* そろえた顔を、まっすぐから何度まわして見せるか（右回り）。
 *   FRAME.upright が true → いつも 0（まっすぐ）
 *   向きを直していない正面の写真 → 撮ったときの向きに近いほう（0 か 180。頭の側から撮った写真はさかさまのまま）
 *   横顔・向きを直した写真 → その向きで見たときの顔の向きにいちばん近い 90度きざみ（寝た横顔は横向きのまま） */
function faceTurn(f, rot, side) {
  if (FRAME.upright) return 0;
  // 元の写真の顔は、まっすぐから -a 度まわって写っている（a は fit がまっすぐに戻す角度）。右に rot 度まわすと rot - a 度
  const seen = (rot || 0) - Math.atan2(f.ai, f.ar) * 180 / Math.PI;
  const steps = rot === null && !side ? [0, 180] : ROTS;
  let best = 0, gap = 999;
  steps.forEach(t => { const g = Math.abs(((seen - t) % 360 + 540) % 360 - 180); if (g < gap) { gap = g; best = t; } });
  return best;
}

/* ---------- 顔の位置を読み取る（管理画面だけが使う） ---------- */
let tasksP = null;
function tasks() {
  if (!tasksP) {
    tasksP = (async () => {
      const vision = await import(LIB + '/vision_bundle.mjs');
      const files = await vision.FilesetResolver.forVisionTasks(LIB + '/wasm');
      // 顔を見つける係（小さい。向きを決めるのに使う）と、顔の点を読む係
      const detector = await vision.FaceDetector.createFromOptions(files, {
        baseOptions: { modelAssetPath: DETECTOR, delegate: 'CPU' }, runningMode: 'IMAGE', minDetectionConfidence: 0.3,
      });
      const make = delegate => vision.FaceLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: MODEL, delegate }, runningMode: 'IMAGE', numFaces: 3,
      });
      let landmarker;
      try { landmarker = await make('GPU'); } catch (e) { landmarker = await make('CPU'); }
      return { detector, landmarker };
    })();
    tasksP.catch(() => { tasksP = null; }); // 読み込めなかったら、次にもう一度ためす
  }
  return tasksP;
}

// 写真を q 度（0・90・180・270）回した絵。長い辺は max まで
function turned(img, w, h, q, max) {
  const s = Math.min(1, max / Math.max(w, h));
  const dw = Math.max(1, Math.round(w * s)), dh = Math.max(1, Math.round(h * s));
  const c = document.createElement('canvas');
  c.width = q % 180 ? dh : dw;
  c.height = q % 180 ? dw : dh;
  const x = c.getContext('2d');
  x.translate(c.width / 2, c.height / 2);
  x.rotate(q * Math.PI / 180);
  x.drawImage(img, -dw / 2, -dh / 2, dw, dh);
  return c;
}

// 写真（img / canvas）から、説明に書く文字を返す。正面で読めなければ横顔として読み、どちらもだめなら 'face:none2'
async function detect(img) {
  const front = await detectFront(img);
  // 正面の点が読めても、大きく横を向いた顔なら、横顔の読み方も試して、読めればそちらを使う。
  // 横顔に無理に正面の点をあてはめると、両目のあいだがとても狭くなる（試した写真: 正面 0.92 以上、横顔 0.46）
  if (front && front.turned < 2.5 && front.width > 0.7) return front.marks;
  let side;
  try {
    side = await detectSide(img);
  } catch (e) {
    // 切り抜きの部品が読み込めなかった（電波など）→「顔なし」とは書かず、あとでもう一度読む
    if (front) return front.marks;
    throw e;
  }
  return side || (front ? front.marks : 'face:none2');
}

async function detectFront(img) {
  const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
  if (!w || !h) throw new Error('empty');
  const { detector, landmarker } = await tasks();
  // 4つの向きで顔の見つけやすさ（0〜1）を比べ、いちばん高い向きで点を読む
  let q = 0, top = -1;
  [0, 90, 180, 270].forEach(t => {
    (detector.detect(turned(img, w, h, t, 640)).detections || []).forEach(d => {
      const sc = d.categories && d.categories[0] ? d.categories[0].score : 0;
      if (sc > top) { top = sc; q = t; }
    });
  });
  const c = turned(img, w, h, q, 1280);
  const faces = (landmarker.detect(c).faceLandmarks || []).filter(f => f && f.length >= 468);
  // いちばん大きく写っている顔（両目のあいだが広い顔）
  const iod = f => Math.hypot((f[263].x + f[362].x - f[33].x - f[133].x) / 2 * c.width, (f[263].y + f[362].y - f[33].y - f[133].y) / 2 * c.height);
  const best = faces.sort((a, b) => iod(b) - iod(a))[0];
  if (!best || iod(best) < 0.04 * Math.min(c.width, c.height)) return null;
  // 横を向いた度合い: 鼻先から左右の目じりまでの長さの比（正面なら 1 くらい、真横に近いほど大きい）
  const len = i => Math.hypot((best[i].x - best[1].x) * c.width, (best[i].y - best[1].y) * c.height);
  const turnedRatio = Math.max(len(33), len(263)) / Math.max(1e-6, Math.min(len(33), len(263)));
  // 顔の横幅の比: 両目のあいだ ÷ 目から口までの縦の長さ（正面なら 1 くらい。真横に近いと目が重なって小さくなる）
  const mid = (i, j) => [(best[i].x + best[j].x) / 2 * c.width, (best[i].y + best[j].y) / 2 * c.height];
  const eye = mid(33, 263), mouth = mid(0, 17);
  const widthRatio = iod(best) / Math.max(1e-6, Math.hypot(mouth[0] - eye[0], mouth[1] - eye[1]));
  // 回した絵の上の点を、元の写真の上の位置（幅・高さに対する割合）に戻す
  const th = q * Math.PI / 180, cos = Math.cos(th), sin = Math.sin(th);
  const dw = q % 180 ? c.height : c.width, dh = q % 180 ? c.width : c.height;
  const p = [];
  POINTS.forEach(([i]) => {
    const a = best[i].x * c.width - c.width / 2, b = best[i].y * c.height - c.height / 2;
    p.push((a * cos + b * sin + dw / 2) / dw, (-a * sin + b * cos + dh / 2) / dh);
  });
  return { marks: format(w / h, p), turned: turnedRatio, width: widthRatio, score: top };
}

/* ---------- 横顔を読む ----------
 * 1. MediaPipe の人の切り抜き（顔の肌・髪・体・服・背景を色分けする部品。初回だけ約16MB）で、顔の肌と背景を分ける
 * 2. 顔の肌のふちのうち、背景に面したいちばん長いひと続き ＝ 横顔の線
 * 3. 線のいちばん出っぱった所 ＝ 鼻先、そのすぐあごの側のへこみ ＝ 鼻の下。額の側は髪のある側
 * 4. 鼻先から額の側へ・あごの側へ決まった長さの線を、標準の横顔の線に重ねる（鼻先と鼻の下で見当をつけ、
 *    近い点どうしを合わせ直すのをくり返す。外れた点の2割は数えない） */
let segP = null;
function segmenter() {
  if (!segP) {
    segP = (async () => {
      const vision = await import(LIB + '/vision_bundle.mjs');
      const files = await vision.FilesetResolver.forVisionTasks(LIB + '/wasm');
      return vision.ImageSegmenter.createFromOptions(files, {
        baseOptions: { modelAssetPath: SEGMENTER, delegate: 'CPU' }, runningMode: 'IMAGE', outputCategoryMask: true, outputConfidenceMasks: false,
      });
    })();
    segP.catch(() => { segP = null; });
  }
  return segP;
}

const CLS = { bg: 0, hair: 1, face: 3 };
async function detectSide(img) {
  const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
  const seg = await segmenter();
  const c = turned(img, w, h, 0, 1024);
  const res = seg.segment(c);
  const cm = res.categoryMask;
  const mw = cm.width, mh = cm.height;
  const mask = Uint8Array.from(cm.getAsUint8Array());
  if (res.close) res.close();
  const r = sideLine(mask, mw, mh);
  if (!r) return null;
  const fitted = sideFit(r);
  if (!fitted) return null;
  // 標準の形の上の4つの点を、写真の上に戻す（重ね方の逆）
  const { a, b, tx, ty } = fitted.s, d = a * a + b * b, p = [];
  for (let i = 0; i < SIDE_KEYS.length; i += 2) {
    const u = SIDE_KEYS[i] - tx, v = SIDE_KEYS[i + 1] - ty;
    p.push((a * u + b * v) / d / mw, (-b * u + a * v) / d / mh);
  }
  return formatSide(w / h, fitted.mirror, p);
}

// 切り抜きの結果から、横顔の線（額 → あごの順の点）と鼻先・鼻の下を取り出す
function sideLine(m, W, H) {
  // 顔の肌のいちばん大きいかたまり
  const lab = new Int32Array(W * H), stack = new Int32Array(W * H);
  let best = 0, bestN = 0, id = 0;
  for (let i = 0; i < W * H; i++) {
    if (m[i] !== CLS.face || lab[i]) continue;
    id++; let n = 0, sp = 0; stack[sp++] = i; lab[i] = id;
    while (sp) {
      const k = stack[--sp]; n++;
      const x = k % W, y = (k - x) / W;
      if (x > 0 && m[k - 1] === CLS.face && !lab[k - 1]) { lab[k - 1] = id; stack[sp++] = k - 1; }
      if (x < W - 1 && m[k + 1] === CLS.face && !lab[k + 1]) { lab[k + 1] = id; stack[sp++] = k + 1; }
      if (y > 0 && m[k - W] === CLS.face && !lab[k - W]) { lab[k - W] = id; stack[sp++] = k - W; }
      if (y < H - 1 && m[k + W] === CLS.face && !lab[k + W]) { lab[k + W] = id; stack[sp++] = k + W; }
    }
    if (n > bestN) { bestN = n; best = id; }
  }
  if (bestN < 0.01 * W * H) return null;
  const size = Math.sqrt(bestN);
  const inF = (x, y) => x >= 0 && y >= 0 && x < W && y < H && lab[y * W + x] === best;
  // かたまりのふちをたどる（ムーアの近傍。時計回り）
  let start = -1;
  for (let i = 0; i < W * H && start < 0; i++) if (lab[i] === best) start = i;
  const DX = [1, 1, 0, -1, -1, -1, 0, 1], DY = [0, 1, 1, 1, 0, -1, -1, -1];
  const C = [];
  let x = start % W, y = (start - x) / W, dir = 7;
  const sx = x, sy = y;
  for (let guard = 0; guard < 4 * W * H; guard++) {
    C.push([x, y]);
    let found = false;
    for (let j = 0; j < 8; j++) {
      const dd = (dir + 6 + j) % 8; // 前の向きから左へ90度もどった所から探す
      const nx = x + DX[dd], ny = y + DY[dd];
      if (inF(nx, ny)) { x = nx; y = ny; dir = dd; found = true; break; }
    }
    if (!found || (x === sx && y === sy && C.length > 2)) break;
  }
  const N = C.length;
  if (N < 20) return null;
  // 背景に面した点（まわり5×5に背景がある）
  const nb = C.map(([px, py]) => {
    for (let yy = py - 2; yy <= py + 2; yy++) for (let xx = px - 2; xx <= px + 2; xx++) {
      if (xx >= 0 && yy >= 0 && xx < W && yy < H && m[yy * W + xx] === CLS.bg) return true;
    }
    return false;
  });
  let s0 = nb.indexOf(false);
  if (s0 < 0) return null;
  let run = 0, rs = 0, bRun = 0, bS = 0;
  for (let j = 1; j <= N; j++) {
    const k = (s0 + j) % N;
    if (nb[k]) { if (!run) rs = k; run++; if (run > bRun) { bRun = run; bS = rs; } } else run = 0;
  }
  if (bRun < size * 0.8) return null;
  let A = [];
  for (let j = 0; j < bRun; j++) A.push(C[(bS + j) % N]);
  // 少しなめらかに
  const k = Math.max(3, Math.floor(size * 0.015)) | 1, hk = (k - 1) / 2;
  A = A.map((_, i) => {
    let ax = 0, ay = 0;
    for (let j = -hk; j <= hk; j++) { const q = A[Math.min(A.length - 1, Math.max(0, i + j))]; ax += q[0]; ay += q[1]; }
    return [ax / k, ay / k];
  });
  const n = A.length;
  // 顔の重心
  let fx = 0, fy = 0;
  for (let i = 0; i < W * H; i++) if (lab[i] === best) { fx += i % W; fy += Math.floor(i / W); }
  fx /= bestN; fy /= bestN;
  // 出っぱり: 前後 r の点を結んだ線から、外（背景の側）へどれだけ出ているか
  const ch = [A[n - 1][0] - A[0][0], A[n - 1][1] - A[0][1]];
  let mx = 0, my = 0;
  A.forEach(q => { mx += q[0]; my += q[1]; });
  mx /= n; my /= n;
  const sgn = ((mx - fx) * -ch[1] + (my - fy) * ch[0]) > 0 ? 1 : -1;
  const prom = rr => {
    rr = Math.max(1, Math.floor(rr));
    const v = new Float64Array(n);
    for (let i = rr; i < n - rr; i++) {
      const a = A[i - rr], b = A[i + rr], tx = b[0] - a[0], ty = b[1] - a[1], l = Math.hypot(tx, ty) || 1;
      v[i] = sgn * ((A[i][0] - (a[0] + b[0]) / 2) * -ty + (A[i][1] - (a[1] + b[1]) / 2) * tx) / l;
    }
    return v;
  };
  const big = prom(size * 0.15);
  let ip = 0;
  for (let i = 1; i < n; i++) if (big[i] > big[ip]) ip = i;
  if (big[ip] < size * 0.02) return null; // はっきりした出っぱり（鼻）がない
  // 額の側 = 髪に近い端
  let hx = 0, hy = 0, hn = 0;
  for (let i = 0; i < W * H; i++) if (m[i] === CLS.hair) { hx += i % W; hy += Math.floor(i / W); hn++; }
  if (!hn) return null;
  hx /= hn; hy /= hn;
  const d0 = Math.hypot(A[0][0] - hx, A[0][1] - hy), d1 = Math.hypot(A[n - 1][0] - hx, A[n - 1][1] - hy);
  const side = d0 < d1 ? -1 : 1; // 額は index が小さい側なら -1
  // 鼻の下: 鼻先からあごの側へ少し進んだ所のへこみ
  const small = prom(size * 0.06);
  const lo = Math.min(n - 1, Math.max(0, ip - side * Math.floor(size * 0.02)));
  const hi = Math.min(n - 1, Math.max(0, ip - side * Math.floor(size * 0.2)));
  let isn = Math.min(lo, hi);
  for (let i = Math.min(lo, hi); i <= Math.max(lo, hi); i++) if (small[i] < small[isn]) isn = i;
  // 線にそった長さ
  const seg = new Float64Array(n);
  for (let i = 1; i < n; i++) seg[i] = seg[i - 1] + Math.hypot(A[i][0] - A[i - 1][0], A[i][1] - A[i - 1][1]);
  // 鼻先から額の側へ SIDE_LO、あごの側へ SIDE_HI（顔の大きさに対する割合）の線を 80 点で取る（額 → あごの順）
  const a0 = seg[ip] + side * SIDE_LO * size, a1 = seg[ip] - side * SIDE_HI * size;
  const X = [];
  for (let j = 0; j < 80; j++) {
    const t = a0 + (a1 - a0) * j / 79;
    if (t < 0 || t > seg[n - 1]) continue;
    let i = 1;
    while (i < n - 1 && seg[i] < t) i++;
    const f = (t - seg[i - 1]) / ((seg[i] - seg[i - 1]) || 1);
    X.push([A[i - 1][0] + (A[i][0] - A[i - 1][0]) * f, A[i - 1][1] + (A[i][1] - A[i - 1][1]) * f]);
  }
  if (X.length < 40) return null;
  // 左向きの顔か（標準は右向き）: 額 → あごの向きと、顔から鼻先への向き
  const dvx = X[X.length - 1][0] - X[0][0], dvy = X[X.length - 1][1] - X[0][1];
  const ox = A[ip][0] - fx, oy = A[ip][1] - fy;
  const mirror = dvx * oy - dvy * ox > 0;
  return { X, P: A[ip], Sn: A[isn], mirror, size };
}

// 横顔の線を標準の横顔の線に重ねる。合わなければ null
function sideFit(r) {
  const sg = r.mirror ? -1 : 1;
  const T = [];
  for (let i = 0; i < SIDE_CURVE.length; i += 2) T.push([SIDE_CURVE[i] * sg, SIDE_CURVE[i + 1]]);
  let ipT = 0;
  T.forEach((q, i) => { if (q[0] * sg > T[ipT][0] * sg) ipT = i; });
  // 見当: 鼻先どうし・鼻の下どうし
  let s = simFit([r.P, r.Sn], [T[ipT], [0, 1]]);
  const near = (z) => {
    let bd = Infinity, bp = null;
    for (let i = 0; i < T.length - 1; i++) {
      const ax = T[i][0], ay = T[i][1], bx = T[i + 1][0] - ax, by = T[i + 1][1] - ay;
      const t = Math.max(0, Math.min(1, ((z[0] - ax) * bx + (z[1] - ay) * by) / (bx * bx + by * by || 1)));
      const px = ax + bx * t, py = ay + by * t, d = (px - z[0]) ** 2 + (py - z[1]) ** 2;
      if (d < bd) { bd = d; bp = [px, py]; }
    }
    return [bp, bd];
  };
  let err = Infinity;
  for (let it = 0; it < 40; it++) {
    const pairs = r.X.map(p => {
      const z = [s.a * p[0] - s.b * p[1] + s.tx, s.b * p[0] + s.a * p[1] + s.ty];
      const [q, d] = near(z);
      return { p, q, d };
    });
    const ds = pairs.map(o => o.d).sort((u, v) => u - v);
    const cut = ds[Math.floor(ds.length * 0.8) - 1];
    const keep = pairs.filter(o => o.d <= cut);
    err = Math.sqrt(keep.reduce((t, o) => t + o.d, 0) / keep.length);
    s = simFit(keep.map(o => o.p), keep.map(o => o.q));
  }
  // 合わないものは使わない（試した横顔47枚: 合わせのずれ 0.005〜0.037、鼻の長さ÷顔の大きさ 0.25〜0.37）
  //   ずれが大きい → 横顔の線の形が合わない（斜めの顔・横顔でない・切り抜きの失敗）
  //   鼻の長さがおかしい → 顔の大きさの読み違い（遠くから撮った写真など）
  const unit = 1 / Math.hypot(s.a, s.b) / r.size;
  if (!(err < 0.04) || !(unit > 0.2 && unit < 0.5)) return null;
  return { s, mirror: r.mirror, err };
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('decode'));
    img.src = src;
  });
}
// スマホで撮った・選んだ写真のファイルから読み取る（向きは写真の中の記録どおりに直して読む）
async function detectFile(file) {
  const url = URL.createObjectURL(file);
  try { return await detect(await loadImage(url)); } finally { URL.revokeObjectURL(url); }
}

window.BeyondFace = { parse, format, fit, usable, render, faceTurn, ROTS, toneLUT, detect, detectFront, detectSide, detectFile, loadImage, warm: tasks, warmSide: segmenter, POINTS, TEMPLATE, FRAME, SIDE_FRAME };
})();
