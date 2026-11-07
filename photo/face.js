/*
 * 美yond 写真ページ: 顔の位置合わせと、アフターの明るさ（お客様ページ index.html と管理画面 admin/index.html の両方で使う）
 *
 * 顔の位置: 管理画面で写真を入れるとき、スタッフのスマホの中で眉・目・鼻・口の位置（20か所）を読み取り、
 *   その写真のドライブの「説明」に数字だけ書いておく。写真そのものはどこにも送らない。
 *   読み取りは Google の MediaPipe（顔の点を478か所見つける仕組み）。管理画面だけが読み込む（初回だけ約15MB）
 * 位置合わせ: お客様ページでは、その数字から「拡大・回転・移動」だけで、どの写真も同じ場所・同じ大きさに顔が来るよう切り抜く。
 *   顔の形は変えない（縦横の伸び縮みやゆがみは入れない）。元の写真はドライブにそのまま残る
 * 明るさ: iPhone の写真アプリの「明るさ」と同じ曲線。Mac の Core Image（写真アプリと同じ仕組み）で、
 *   目盛り -100〜100 を 10 きざみで測った表を持ち、間はなめらかにつなぐ（ずれは 256 段階で1段まで）
 */
(function () {
'use strict';

const LIB = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

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
const FRAME = { ratio: 4 / 3, cx: 0.5, cy: 0.42 * 4 / 3, iod: 0.30 };
const FILL = '#E4EDEC'; // 元の写真の外にはみ出たところの色（ページの写真枠と同じ）

/* ---------- 説明に書く文字（face:v1;a=横÷縦;p=x,y,x,y…  位置は写真の幅・高さに対する割合） ---------- */
function parse(s) {
  if (typeof s !== 'string' || s.indexOf('face:') !== 0) return null;
  if (s === 'face:none') return 'none';
  const parts = s.slice(5).split(';');
  if (parts[0] !== 'v1') return null; // 書き方が古い・新しい → もう一度読み取る
  const kv = {};
  parts.slice(1).forEach(t => { const i = t.indexOf('='); if (i > 0) kv[t.slice(0, i)] = t.slice(i + 1); });
  const p = String(kv.p || '').split(',').map(Number);
  const aspect = Number(kv.a);
  if (p.length !== POINTS.length * 2 || p.some(v => !isFinite(v)) || !(aspect > 0)) return null;
  return { aspect, p };
}
function format(aspect, p) {
  return 'face:v1;a=' + aspect.toFixed(4) + ';p=' + p.map(v => v.toFixed(4)).join(',');
}

/* ---------- 合わせ方: 写真の点を標準の配置に、拡大・回転・移動だけで重ねる（重みつきの最小二乗） ---------- */
function fit(m, w, h) {
  let sw = 0, zx = 0, zy = 0, tx = 0, ty = 0;
  const X = [], Y = [];
  POINTS.forEach(([, wt], i) => {
    X[i] = m.p[i * 2] * w; Y[i] = m.p[i * 2 + 1] * h;
    sw += wt; zx += wt * X[i]; zy += wt * Y[i]; tx += wt * TEMPLATE[i * 2]; ty += wt * TEMPLATE[i * 2 + 1];
  });
  zx /= sw; zy /= sw; tx /= sw; ty /= sw;
  let nr = 0, ni = 0, d = 0;
  POINTS.forEach(([, wt], i) => {
    const x = X[i] - zx, y = Y[i] - zy, u = TEMPLATE[i * 2] - tx, v = TEMPLATE[i * 2 + 1] - ty;
    nr += wt * (u * x + v * y); ni += wt * (v * x - u * y); d += wt * (x * x + y * y);
  });
  const ar = nr / d, ai = ni / d;
  return { ar, ai, bx: tx - (ar * zx - ai * zy), by: ty - (ai * zx + ar * zy) };
}

// 説明の数字がこの写真のものとして使えるか（縦横の比が合わないときは、向きが変わったとみなして使わない）
function usable(m, w, h) {
  return !!(m && m !== 'none' && m.p && w && h && Math.abs(m.aspect - w / h) < 0.03);
}

/* ---------- 明るさ（iPhone の「明るさ」の目盛り -100〜100）の表 ---------- */
const TONE_B64 = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEBAQEBAQEBAQEBAQICAgICAgIDAwMDBAQEBAUFBQYGBgcHCAgJCQoKCwwMDQ0ODw8QERESExQUFRYWFxgZGhobHB0eHx8gISIjJCUmJycoKSorLC0uLzAxMzQ1Njc4OTo7PT4/QEFCREVGR0lKS01OT1BSU1VWV1laXF1eYGFjZGZnaWpsbW9xcnR1d3l6fH5/gYOFhoiKjI2PkZOVl5ianJ6goqSmqKqsrrCytLa4ury+wMPFx8nLztDS1NbZ293g4uTn6evu8PP1+Pr9/wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAQEBAQEBAQEBAQECAgICAgICAwMDAwQEBAQFBQUGBgYHBwgICQkKCgsMDA0ODg8PEBEREhMUFBUWFxcYGRoaGxwdHh4fICEiIyQkJSYnKCkqKywtLi8wMTIzNDU2Nzg5Ozw9Pj9AQUNERUZHSUpLTE1PUFFTVFVXWFlbXF1fYGJjZGZnaWpsbW9wcnN1dnh6e31+gIKDhYeIioyNj5GTlJaYmpudn6GjpaeoqqyusLK0tri6vL7AwsTGyMrMztDS1NfZ293f4eTm6Ort7/Hz9vj6/f8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQEBAQEBAQEBAQEBAgICAgICAwMDAwMEBAQEBQUFBgYHBwgICQkKCgsLDA0NDg8PEBAREhMTFBUVFhcYGBkaGxscHR4fICAhIiMkJSYmJygpKissLS4vMDEyMzQ1Njc4OTo7PD4/QEFCQ0RFR0hJSktNTk9QUVNUVVdYWVpcXV5gYWNkZWdoaWtsbm9xcnR1d3h6e31+gIGDhIaIiYuMjpCRk5WWmJqbnZ+hoqSmqKmrra+xsrS2uLq8vr/Bw8XHycvNz9HT1dfZ293f4ePl5+ns7vDy9Pb4+/3/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAQEBAQEBAQEBAQICAgICAgMDAwMEBAQEBQUFBgYHBwcICAkKCgsLDA0NDg4PEBAREhMTFBUVFhcYGBkaGxscHR4fHyAhIiMkJSUmJygpKissLS4vMDExMjM0NTY3OTo7PD0+P0BBQkNERUdISUpLTE5PUFFSVFVWV1haW1xeX2BhY2RlZ2hpa2xtb3Byc3R2d3l6fH1/gIKDhYaIiYuMjo+RkpSWl5manJ6foaKkpqepq62usLKztbe5ury+wMHDxcfJy8zO0NLU1tjZ293f4ePl5+nr7e/x8/X3+fv9/wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQEBAQEBAQEBAQICAgICAgMDAwMEBAQEBQUFBgYHBwgICQkKCgsMDA0NDg8PEBEREhMUFBUWFhcYGRkaGxwdHR4fICEhIiMkJSYnKCgpKissLS4vMDEyMzQ1Njc4OTo7PD0+P0BBQkNERUZHSEpLTE1OT1BSU1RVVldZWltcXl9gYWJkZWZoaWprbW5vcXJzdXZ3eXp8fX6AgYOEhYeIiouNjpCRk5SWl5manJ2foKKjpaaoqqutrrCys7W2uLq7vb/AwsTFx8nLzM7Q0dPV19na3N7g4ePl5+nq7O7w8vT29/n7/f8AAAAAAAAAAAAAAAAAAAAAAAAAAAEBAQEBAQEBAQECAgICAgMDAwMEBAQFBQUGBgYHBwgICQoKCwsMDQ0ODw8QERISExQUFRYXFxgZGhobHB0eHh8gISIiIyQlJicoKCkqKywtLi8wMTIzMzQ1Njc4OTo7PD0+P0BBQkRFRkdISUpLTE1OT1FSU1RVVldZWltcXV5gYWJjZGZnaGlqbG1ub3Fyc3V2d3h6e3x+f4CCg4SGh4iKi42Oj5GSlJWWmJmbnJ6foKKjpaaoqausrq+xsrS1t7i6u72/wMLDxcbIysvNztDS09XW2Nrb3d/g4uTl5+nq7O7v8fP19vj6/P3/AAAAAAAAAAAAAAAAAAABAQEBAQEBAQECAgICAgMDAwMEBAQFBQYGBgcHCAkJCgoLDAwNDg4PEBAREhMTFBUWFhcYGRkaGxwdHR4fICEhIiMkJSYnJygpKissLS4vMDAxMjM0NTY3ODk6Ozw9Pj9AQUJDREVGR0hJSktMTU5QUVJTVFVWV1hZW1xdXl9gYWJkZWZnaGlrbG1ub3Fyc3R2d3h5enx9fn+BgoOFhoeIiouMjo+QkZOUlZeYmZucnZ+goqOkpqeoqqutrq+xsrS1tri5u7y+v8DCw8XGyMnLzM7P0dLU1dfY2tvd3uDh4+Tm5+nr7O7v8fL09ff5+vz9/wAAAAAAAAAAAQEBAQEBAQECAgICAgMDAwQEBAUFBgYHBwgICQkKCwwMDQ4ODxARERITFBQVFhcXGBkaGxscHR4fICAhIiMkJSUmJygpKissLS0uLzAxMjM0NTY3ODk6Ozw9Pj4/QEFCQ0RFRkdISktMTU5PUFFSU1RVVldYWVpbXV5fYGFiY2RlZmhpamtsbW5vcXJzdHV2eHl6e3x9f4CBgoOFhoeIiYuMjY6QkZKTlJaXmJmbnJ2eoKGipKWmp6mqq62ur7Cys7S2t7i6u7y+v8DCw8TGx8jKy8zOz9HS09XW19na3N3e4OHi5OXn6Orr7O7v8fLz9fb4+fv8/v8AAAAAAQEBAQECAgICAwMDBAQEBQUGBgcHCAkJCgsMDA0ODw8QERITExQVFhcXGBkaGxwcHR4fICEiIiMkJSYnKCkqKyssLS4vMDEyMzQ1Njc4ODk6Ozw9Pj9AQUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVtcXV5fYGFiY2RlZmdoaWpsbW5vcHFyc3R1dnh5ent8fX5/gIKDhIWGh4iKi4yNjo+QkpOUlZaXmZqbnJ2eoKGio6Slp6ipqqutrq+wsbO0tba3ubq7vL2/wMHCxMXGx8jKy8zNz9DR0tTV1tfZ2tvc3t/g4ePk5efo6ers7e7w8fLz9fb3+fr7/P7/AAABAQECAgMDBAQFBQYGBwgICQoLDA0NDg8QERITFBQVFhcYGRobHBwdHh8gISIjJCUmJycoKSorLC0uLzAxMjM0NTY3ODk6Ojs8PT4/QEFCQ0RFRkdISUpLTE1OT1BRUlNUVVZXWFlaW1xdXl9gYWJjZWZnaGlqa2xtbm9wcXJzdHV2d3h5ent8fn+AgYKDhIWGh4iJiouMjY+QkZKTlJWWl5iZmpudnp+goaKjpKWmp6mqq6ytrq+wsbK0tba3uLm6u7y9v8DBwsPExcbHycrLzM3Oz9DS09TV1tfY2dvc3d7f4OHi5OXm5+jp6uzt7u/w8fL09fb3+Pn6/P3+/wABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4fICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj9AQUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVpbXF1eX2BhYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5ent8fX5/gIGCg4SFhoeIiYqLjI2Oj5CRkpOUlZaXmJmam5ydnp+goaKjpKWmp6ipqqusra6vsLGys7S1tre4ubq7vL2+v8DBwsPExcbHyMnKy8zNzs/Q0dLT1NXW19jZ2tvc3d7f4OHi4+Tl5ufo6err7O3u7/Dx8vP09fb3+Pn6+/z9/v8AAQIDBQYHCAkKCwwNDg8QERITFBUWFxgZGhwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0+P0BBQkNERUZHSElKS01OT1BRUlNUVVZXWFlaW1xdXl9gYWJjZGVmZ2hpamtsbW5vcHFyc3R1dnh5ent8fX5/gIGCg4SFhoeIiYqLjI2Oj5CRkpOUlZaXmJmanJ2en6ChoqOkpaanqKmqq6ytrq+wsbKztLW2t7i5uru8vb6/wMHCw8TFxsfIycrLzM3Oz9DR0tPU1dbX2Nna29zd3t/g4eLj5OXm5+jp6uvr7O3u7/Dx8vP09PX29/j5+fr7/Pz9/v7/AAIDBAYHCAoLDA0ODxASExQVFhcYGRobHB0eHyEiIyQlJicoKSorLC0uLzEyMzQ1Njc4OTo7PD0+P0BCQ0RFRkdISUpLTE1OT1BRUlNVVldYWVpbXF1eX2BhYmNkZWZoaWprbG1ub3BxcnN0dXZ3eHl6e31+f4CBgoOEhYaHiImKi4yNjo+QkZOUlZaXmJmam5ydnp+goaKjpKWmp6ipqqusra6vsbKztLW2t7i5uru8vb6/wMHCw8TFxsfIycrLzM3Oz8/Q0dLT1NXW19jZ2tvc3d7f4ODh4uPk5ebn6Ojp6uvs7e3u7/Dx8vLz9PX19vf4+Pn6+vv7/P39/v7//wACBAYHCQsMDQ8QERITFBYXGBkaGxweHyAhIiMkJScoKSorLC0uLzEyMzQ1Njc4OTs8PT4/QEFCQ0RGR0hJSktMTU5PUVJTVFVWV1hZWltdXl9gYWJjZGVmZ2lqa2xtbm9wcXJzdHV3eHl6e3x9fn+AgYKDhIaHiImKi4yNjo+QkZKTlJWWmJmam5ydnp+goaKjpKWmp6ipqqusra6vsLGys7S1tre4ubq7vL2+v8DBwsPExcbHyMnKy8zNzs/Q0dLS09TV1tfY2drb3Nzd3t/g4eLi4+Tl5ufn6Onq6+vs7e7u7/Dx8fLz8/T19fb39/j5+fr6+/v8/P39/v7+//8AAwUICgwODxESExQWFxgZGhwdHh8gISMkJSYnKCorLC0uLzEyMzQ1Njg5Ojs8PT5AQUJDREVGSElKS0xNTlBRUlNUVVZYWVpbXF1eX2FiY2RlZmdoaWtsbW5vcHFyc3V2d3h5ent8fX6AgYKDhIWGh4iJiouNjo+QkZKTlJWWl5iZmpucnp+goaKjpKWmp6ipqqusra6vsLGys7S1tre4ubq7vL2+v8DBwsPExcbHyMjJysvMzc7P0NHS09PU1dbX2Nna2tvc3d7f3+Dh4uPj5OXm5+fo6erq6+zs7e7u7/Dw8fLy8/T09fX29/f4+Pn5+vr7+/z8/P39/v7+/v//AAUICw0PERMUFhcYGhscHR8gISIjJSYnKCorLC0uMDEyMzQ2Nzg5Ojw9Pj9AQUNERUZHSUpLTE1OUFFSU1RVV1hZWltcXl9gYWJjZGZnaGlqa2xub3BxcnN0dnd4eXp7fH1+gIGCg4SFhoeIiouMjY6PkJGSk5SWl5iZmpucnZ6foKGio6Slpqepqqusra6vsLGys7S1tre4ubq7vL29vr/AwcLDxMXGx8jJysvMzM3Oz9DR0tPT1NXW19jZ2drb3N3d3t/g4eHi4+Tk5ebn5+jp6err6+zt7e7v7/Dw8fLy8/P09PX29vf3+Pj4+fn6+vv7+/z8/P39/f7+/v///wAIDA8SFBYYGRscHR4gISIjJSYnKCorLC0vMDEyNDU2Nzg6Ozw9P0BBQkNFRkdISUtMTU5PUVJTVFVXWFlaW11eX2BhYmRlZmdoaWtsbW5vcHJzdHV2d3l6e3x9fn+BgoOEhYaHiIqLjI2Oj5CRkpOVlpeYmZqbnJ2en6Cho6SlpqeoqaqrrK2ur7CxsrO0tba3uLm6u7y9vr/AwcLCw8TFxsfIycrLzM3Nzs/Q0dLT09TV1tfY2Nna29zc3d7f3+Dh4uLj5OTl5ufn6Onp6urr7Ozt7u7v7/Dw8fLy8/P09PX19vb39/f4+Pn5+fr6+/v7/Pz8/f39/f7+/v7///8ADRIVGBocHR8gIiMkJicoKSssLS4wMTIzNTY3ODk7PD0+QEFCQ0RGR0hJS0xNTk9RUlNUVVdYWVpcXV5fYGJjZGVmZ2lqa2xtb3BxcnN0dnd4eXp7fX5/gIGChIWGh4iJiouNjo+QkZKTlJaXmJmam5ydnp+goaOkpaanqKmqq6ytrq+wsbKztLW2t7i5uru8vb6/wMHCw8PExcbHyMnKy8zMzc7P0NHS0tPU1dbX19jZ2trb3N3e3t/g4OHi4+Pk5eXm5+fo6enq6uvs7O3t7u7v8PDx8fLy8/P09PT19fb29/f3+Pj5+fn6+vr7+/v7/Pz8/f39/f7+/v7+////ABUZHB8hIyUmKCkqKy0uLzAyMzQ1Njg5Ojs9Pj9AQUNERUZHSUpLTE5PUFFSVFVWV1haW1xdXmBhYmNkZmdoaWpsbW5vcHFzdHV2d3l6e3x9foCBgoOEhYaIiYqLjI2OkJGSk5SVlpeZmpucnZ6foKGio6Smp6ipqqusra6vsLGys7S1tre4ubq7vL2+v8DBwsLDxMXGx8jJysvMzM3Oz9DR0dLT1NXW1tfY2dna29zc3d7f3+Dh4eLj4+Tl5ebn5+jp6erq6+vs7e3u7u/v8PDx8fLy8/P09PT19fb29vf39/j4+Pn5+fr6+vv7+/v8/Pz8/f39/f3+/v7+/v///wAeIyYoKiwuLzEyMzQ1Nzg5Ojs8Pj9AQUJDRUZHSElKTE1OT1BRU1RVVldYWltcXV5fYWJjZGVnaGlqa2xub3BxcnN1dnd4eXp8fX5/gIGDhIWGh4iKi4yNjo+QkZOUlZaXmJmam52en6ChoqOkpaanqKmrrK2ur7CxsrO0tba3uLm6u7y9vr/AwMHCw8TFxsfIycrKy8zNzs/Q0NHS09TV1dbX2NjZ2tvb3N3e3t/g4OHi4uPk5OXm5ufn6Onp6urr6+zt7e7u7+/w8PHx8fLy8/P09PT19fb29vf39/j4+Pn5+fn6+vr6+/v7+/z8/Pz9/f39/f3+/v7+/v////8AJywvMTM1Njg5Ojs8PT4/QUJDREVGR0hJS0xNTk9QUVJTVVZXWFlaW11eX2BhYmNlZmdoaWprbW5vcHFydHV2d3h5e3x9fn+AgoOEhYaHiIqLjI2Oj5CSk5SVlpeYmZqcnZ6foKGio6Slpqipqqusra6vsLGys7S1tre4ubq7vL2+v8DBwsPExMXGx8jJysvMzM3Oz9DR0tLT1NXW1tfY2dna29zc3d7e3+Dh4eLi4+Tk5ebm5+fo6enq6uvr7Ozt7e7u7+/w8PHx8vLy8/P09PT19fX29vb39/f4+Pj5+fn5+vr6+vv7+/v7/Pz8/Pz9/f39/f3+/v7+/v7/////';
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

/* ---------- 画面に出す絵を作る ----------
 * marks があれば顔の位置で切り抜いた 横3:縦4 の絵、なければ元の写真のまま（長い辺 1600 まで）。bright は明るさの目盛り（0 なら何もしない） */
function render(img, opts) {
  opts = opts || {};
  const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
  const canvas = opts.canvas || document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const aligned = usable(opts.marks, w, h);
  if (aligned) {
    const f = fit(opts.marks, w, h);
    // 枠の幅が元の写真の何ピクセル分か。それより細かくしても見た目は変わらないので、絵の大きさはそこまで（480〜1080）
    const crop = 1 / (FRAME.iod * Math.hypot(f.ar, f.ai));
    const W = Math.round(Math.min(1080, Math.max(480, crop)));
    canvas.width = W;
    canvas.height = Math.round(W * FRAME.ratio);
    ctx.fillStyle = FILL;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const k = W * FRAME.iod;
    ctx.setTransform(k * f.ar, k * f.ai, -k * f.ai, k * f.ar, k * f.bx + W * FRAME.cx, k * f.by + W * FRAME.cy);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, 0, 0, w, h);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  } else {
    const s = Math.min(1, 1600 / Math.max(w, h));
    canvas.width = Math.max(1, Math.round(w * s));
    canvas.height = Math.max(1, Math.round(h * s));
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  }
  if (opts.bright) {
    try { tone(ctx, canvas.width, canvas.height, opts.bright); } catch (e) { /* 読めない絵のときは明るさだけあきらめる */ }
  }
  return { canvas, aligned };
}

/* ---------- 顔の位置を読み取る（管理画面だけが使う） ---------- */
let landmarkerP = null;
function landmarker() {
  if (!landmarkerP) {
    landmarkerP = (async () => {
      const vision = await import(LIB + '/vision_bundle.mjs');
      const files = await vision.FilesetResolver.forVisionTasks(LIB + '/wasm');
      const make = delegate => vision.FaceLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: MODEL, delegate }, runningMode: 'IMAGE', numFaces: 3,
      });
      try { return await make('GPU'); } catch (e) { return await make('CPU'); }
    })();
    landmarkerP.catch(() => { landmarkerP = null; }); // 読み込めなかったら、次にもう一度ためす
  }
  return landmarkerP;
}

// 写真（img / canvas）から、説明に書く文字を返す。顔が見つからなければ 'face:none'
async function detect(img) {
  const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
  if (!w || !h) throw new Error('empty');
  const s = Math.min(1, 1280 / Math.max(w, h));
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(w * s));
  c.height = Math.max(1, Math.round(h * s));
  c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
  const lm = await landmarker();
  const faces = (lm.detect(c).faceLandmarks || []).filter(f => f && f.length >= 468);
  // いちばん大きく写っている顔（両目のあいだが広い顔）
  const iod = f => Math.hypot((f[263].x + f[362].x - f[33].x - f[133].x) / 2 * c.width, (f[263].y + f[362].y - f[33].y - f[133].y) / 2 * c.height);
  const best = faces.sort((a, b) => iod(b) - iod(a))[0];
  if (!best || iod(best) < 0.04 * Math.min(c.width, c.height)) return 'face:none';
  const p = [];
  POINTS.forEach(([i]) => { p.push(best[i].x, best[i].y); });
  return format(w / h, p);
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

window.BeyondFace = { parse, format, fit, usable, render, toneLUT, detect, detectFile, loadImage, warm: landmarker, POINTS, TEMPLATE, FRAME };
})();
