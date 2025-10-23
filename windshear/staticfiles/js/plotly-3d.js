// 颜色与中文映射
console.log('JS loaded, binding form...');
const colorMap = {0:'#ffffff',1:'#00ff00',2:'#ffff00',3:'#ff8800',4:'#ff0000'};
const nameMap  = {0:'无',1:'轻度',2:'中度',3:'强烈',4:'严重'};

// 绑定上传事件
document.getElementById('uploadForm').addEventListener('submit', async (e)=>{
  e.preventDefault();
  const formData = new FormData(e.target);
  const res = await fetch('/api/predict/', {method:'POST', body:formData});
  const json = await res.json();
  if(!res.ok){ alert(json.error||'预测失败'); return; }
  draw3D(json.points);
});

// 3D 绘图
function draw3D(points){
  const traces = [];
  for(let lbl=0;lbl<5;lbl++){
    const sub = points.filter(p=>p.label===lbl);
    if(sub.length===0) continue;
    traces.push({
      x: sub.map(p=>p.x),
      y: sub.map(p=>p.y),
      z: sub.map(p=>p.z),
      mode:'markers',
      type:'scatter3d',
      name:`等级 ${lbl} (${nameMap[lbl]})`,
      marker:{size:3, color:colorMap[lbl]}
    });
  }
  const layout = {
    title:'风切变强度 3D 分布',
    scene:{xaxis:{title:'X (m)'}, yaxis:{title:'Y (m)'}, zaxis:{title:'Z (m)'}}
  };
  Plotly.newPlot('plot3d', traces, layout);
}
