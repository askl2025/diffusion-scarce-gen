'use strict';
const $ = (s) => document.querySelector(s);
let selectedFile = null, previewUrl = null, current = null, tab = 'result', ready = false;
let promptInitialized = false, submitting = false;
let lastDone = null, jobs = [], polling = false;
let prompts = {}, previewImage = null, placementValid = true;
let profiles = {}, placementDrafts = {}, placementChosen = false, pendingDefaults = null;
let reviewImage = null, reviewBoxes = [], reviewJobId = null, dragStart = null, dragEnd = null;
const terminal = (j) => ['done','error','cancelled'].includes(j.status);
const labels = {queued:'等待中',masking:'生成遮罩',generating:'正在生图',done:'已完成',error:'失败',cancelled:'已取消'};
const rectangleType = (type=$('#object-type').value) => ['warning_sign','pedestrian'].includes(type);
const placementMode = () => rectangleType() ? 'manual' : $('#placement-mode').value;
function placementValues(type=$('#object-type').value){
  const values={x:Number($('#place-x').value)/100,y:Number($('#place-y').value)/100,width:Number($('#place-width').value)/100};
  if(rectangleType(type))values.height=Number($('#place-height').value)/100;
  return values;
}
function applyPlacementDefaults(type){
  const values=placementDrafts[type]||profiles[type]?.placement_defaults;
  if(!values)return false;
  for(const [key,id] of [['x','place-x'],['y','place-y'],['width','place-width'],['height','place-height']]){
    if(values[key]!==undefined)$('#'+id).value=(values[key]*100).toFixed(1);
  }
  return true;
}

async function api(path, options) {
  const r = await fetch(path, options);
  const data = await r.json().catch(() => ({error:'服务器响应异常，请稍后重试'}));
  if (!r.ok) throw new Error(data.error || '请求失败');
  return data;
}
function error(text='') { $('#error').textContent=text; $('#error').hidden=!text; }
function buttonState() {
  const busy = current && !terminal(current);
  $('#start').disabled = !ready || !selectedFile || busy || submitting || !placementValid || (rectangleType()&&!placementChosen);
  $('#start span').textContent = submitting ? '正在上传图片' : busy ? '处理中，请稍候' : '启动自动生成';
}
function choose(file) {
  if (!file) return;
  if (!['image/jpeg','image/png','image/webp'].includes(file.type)) return error('请选择 JPG、PNG 或 WebP 图片。');
  if (file.size>12*1024*1024) return error('图片不能超过 12 MB。');
  error(); selectedFile=file; placementChosen=false;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl=URL.createObjectURL(file);
  previewImage = new Image();
  previewImage.onload=()=>{drawPlacement();buttonState();}; previewImage.src=previewUrl;
  $('#upload-preview').src=previewUrl; $('#upload-preview').hidden=false;
  $('#upload-empty').hidden=true; $('#replace-hint').hidden=false;
  $('#file-meta').textContent=`${file.name} · ${(file.size/1024/1024).toFixed(2)} MB`;
  buttonState();
}
$('#dropzone').onclick=()=>$('#file').click();
$('#dropzone').onkeydown=(e)=>{if(['Enter',' '].includes(e.key)){e.preventDefault();$('#file').click();}};
$('#file').onchange=(e)=>choose(e.target.files[0]);
for (const type of ['dragenter','dragover']) $('#dropzone').addEventListener(type,e=>{e.preventDefault();$('#dropzone').classList.add('dragging');});
for (const type of ['dragleave','drop']) $('#dropzone').addEventListener(type,e=>{e.preventDefault();$('#dropzone').classList.remove('dragging');});
$('#dropzone').addEventListener('drop',e=>choose(e.dataTransfer.files[0]));

async function checkHealth(resetPrompt=false) {
  const type=$('#object-type').value, mode=placementMode(), protect=$('#protect-original').checked;
  try {
    const health=await api('api/health?object_type='+type+'&placement_mode='+mode+'&protect_original='+(protect?'1':'0'));
    if(type!==$('#object-type').value || mode!==placementMode()||protect!==$('#protect-original').checked)return;
    profiles=health.profiles||{};
    if(pendingDefaults===type&&!placementChosen){applyPlacementDefaults(type);pendingDefaults=null;}
    ready=health.ready;
    $('#service').className='service '+(ready?'ready':'failed');
    $('#service span').textContent=ready?'生成服务已就绪':'服务未就绪';
    $('#service').title=health.missing.join('、');
    if(resetPrompt || !promptInitialized) { $('#prompt').value=prompts[type]||health.default_prompt; promptInitialized=true; }
  } catch(e) { ready=false; $('#service').className='service failed'; $('#service span').textContent='服务连接中断'; }
  placementControls();
}
function setCanvas() {
  const files=current?.files||{};
  let key=tab;
  if(!files[key] && tab==='result') key=files.overlay?'overlay':files.input?'input':null;
  const src=key?files[key]:null;
  $('#canvas-empty').hidden=!!src; $('#canvas-image').hidden=!src;
  $('#canvas-badge').hidden=!src; $('#zoom').hidden=!src;
  if(src) {
    if(!$('#canvas-image').src.endsWith(src)) $('#canvas-image').src=src;
    $('#canvas-badge').textContent={input:'本次处理原图',overlay:'绿色区域将被重绘','protection-overlay':'青色为已识别并保护的原有车辆和行人',result:'生成结果'}[key];
  }
  document.querySelectorAll('[data-tab]').forEach(b=>{
    b.setAttribute('aria-selected',String(b.dataset.tab===tab));
    b.disabled=!!current && !files[b.dataset.tab] && b.dataset.tab!=='result';
  });
}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{tab=b.dataset.tab;setCanvas();});
function showJob(job) {
  current=job; localStorage.setItem('roadlab-job',job.id);
  $('#status-label').textContent=job.message;
  $('#progress').style.width=job.progress+'%'; $('#percent').textContent=job.progress+'%';
  $('#run-status').classList.toggle('busy',!terminal(job));
  $('#cancel').hidden=terminal(job);
  const stage={queued:0,masking:1,generating:2,done:3}[job.status]??-1;
  document.querySelectorAll('[data-stage]').forEach(el=>{
    const n=Number(el.dataset.stage); el.className=n<stage?'complete':n===stage?'active':'';
  });
  for(const name of ['mask','result']) {
    const link=$('#download-'+name), src=job.files?.[name];
    link.hidden=!src; if(src) {link.href=src+'?download=1';link.download='roadlab_'+name+'.png';}
  }
  $('#download-bundle').hidden=job.status!=='done'; $('#download-bundle').href=job.bundle_url;
  $('#result-meta').textContent=job.status==='done'
    ?`${job.object_label||'交通锥 / 路障'} · ${job.size.join(' × ')} px · ${job.settings.steps} 步 · 用时 ${job.elapsed}s`
    :job.lane_count!==null?`检测到 ${job.lane_count} 条车道 · ${job.size.join(' × ')} px`:'所有步骤自动衔接 · 结果保留在服务器';
  if(job.status==='error') error(job.message);
  setCanvas(); buttonState();
  loadReview(job);
  if(terminal(job) && lastDone!==job.id+job.status){lastDone=job.id+job.status;loadHistory(false);}
}
$('#start').onclick=async()=>{
  if(!selectedFile) return;
  error(); submitting=true; buttonState();
  const settings={object_type:$('#object-type').value,prompt:$('#prompt').value,steps:$('#steps').value,max_side:$('#max-side').value,expand:$('#expand').value,seed:$('#seed').value,
    placement_mode:placementMode(),placement:placementValues(),protect_original:$('#protect-original').checked};
  const form=new FormData(); form.append('image',selectedFile);form.append('settings',JSON.stringify(settings));
  try {const job=await api('api/jobs',{method:'POST',body:form});tab='result';showJob(job);await loadHistory(false);}
  catch(e){error(e.message);}finally{submitting=false;buttonState();}
};
$('#cancel').onclick=async()=>{
  if(!current)return;
  $('#cancel').disabled=true;
  try{showJob(await api('api/jobs/'+current.id+'/cancel',{method:'POST'}));}
  catch(e){error(e.message);}finally{$('#cancel').disabled=false;}
};
async function poll(){
  if(polling||!current||terminal(current))return;
  polling=true;const id=current.id;
  try{const job=await api('api/jobs/'+id);if(current?.id===id)showJob(job);}
  catch(e){$('#status-label').textContent='连接暂时中断，正在重试；后台任务会继续运行';}
  finally{polling=false;}
}
async function loadHistory(restore=false){
  try{
    jobs=await api('api/jobs');$('#history-count').textContent=jobs.length;
    const list=$('#history-list');list.replaceChildren();
    if(!jobs.length){const p=document.createElement('p');p.className='history-empty';p.textContent='还没有生成记录。第一张新场景，从上方开始。';list.append(p);}
    for(const job of jobs){
      const card=document.createElement('button');card.className='history-card'+(job.id===current?.id?' selected':'');
      const img=document.createElement('img');img.src=job.files.result||job.files.overlay||job.files.input;img.alt='实验图片';img.loading='lazy';
      const info=document.createElement('div');info.className='card-info';
      const label=document.createElement('span');label.textContent=(job.object_label||'交通锥 / 路障')+' · '+(labels[job.status]||job.status);if(job.status==='error')label.className='state-error';
      const time=document.createElement('small');time.textContent=new Date(job.created_at*1000).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});
      info.append(label,time);card.append(img,info);card.onclick=()=>{error();tab='result';showJob(job);loadHistory(false);};list.append(card);
    }
    if(restore){const id=localStorage.getItem('roadlab-job');const j=jobs.find(j=>j.id===id)||jobs[0];if(j)showJob(j);}
  }catch(e){error(e.message);}
}
$('#refresh').onclick=()=>{loadHistory(false);checkHealth();};
$('#zoom').onclick=()=>{$('#large-image').src=$('#canvas-image').src;$('#viewer').showModal();};
$('#close-viewer').onclick=()=>$('#viewer').close();
$('#viewer').onclick=e=>{if(e.target===$('#viewer'))$('#viewer').close();};
function drawPlacement(){
  const rectangular=rectangleType();
  const manual=rectangular||($('#object-type').value==='dustbin' && placementMode()==='manual');
  $('#placement-canvas').hidden=!manual||!previewImage?.complete;
  placementValid=true;
  if(!manual||!previewImage?.naturalWidth)return;
  const c=$('#placement-canvas'), scale=Math.min(1,800/previewImage.naturalWidth);
  c.width=Math.round(previewImage.naturalWidth*scale);c.height=Math.round(previewImage.naturalHeight*scale);
  const ctx=c.getContext('2d');ctx.drawImage(previewImage,0,0,c.width,c.height);
  const x=Number($('#place-x').value)/100*c.width,y=Number($('#place-y').value)/100*c.height;
  const w=Number($('#place-width').value)/100*c.width,h=rectangular?Number($('#place-height').value)/100*c.height:w*1.65,pad=rectangular?0:Math.max(4,c.width/128);
  placementValid=Number.isFinite(x+y+w+h)&&x>=c.width*.05&&x<=c.width*.95&&y>=c.height*.25&&y<=c.height*.96&&w>=c.width*.035&&w<=c.width*.18&&(!rectangular||(h>=c.height*.08&&h<=c.height*.70))&&x-w/2-pad>=0&&x+w/2+pad<c.width&&y-h-pad>=0&&y+pad*1.5<c.height;
  ctx.fillStyle=placementValid?'#a4ef5b55':'#ff776655';ctx.fillRect(x-w/2-pad,y-h-pad,w+pad*2,h+pad*2);
  ctx.strokeStyle=placementValid?'#d2f56b':'#ff8877';ctx.lineWidth=2;ctx.strokeRect(x-w/2,y-h,w,h);
  ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.stroke();
  $('#placement-note').textContent=!placementValid?'区域超出图片，请调整位置或缩小编辑区域。':rectangular&&!placementChosen?'请点击预览选择底部位置，再调整编辑区域宽高。':rectangular?'绿色框是允许重绘的范围，可继续调整宽高。':'点击图片设置底部接地点；绿色框是允许重绘区域。';
}
function placementControls(){
  const type=$('#object-type').value, rectangular=rectangleType(),dustbin=type==='dustbin', manual=placementMode()==='manual';
  $('#placement-controls').hidden=!dustbin&&!rectangular;$('#manual-controls').hidden=!manual;$('#expand-field').hidden=type!=='traffic_cone';
  $('#placement-mode-field').hidden=rectangular;$('#place-height-field').hidden=!rectangular;
  $('#native-speed').hidden=!rectangular;
  $('#generation-preset').value=['20','30'].includes($('#steps').value)?$('#steps').value:'custom';
  $('#place-width-label').textContent=rectangular?'编辑区域宽 / 图片宽度 %':'桶宽 / 图片宽度 %';
  $('#placement-context').hidden=!rectangular;$('#placement-context').textContent=profiles[type]?.placement_note||'请手动选择符合场景的编辑区域。';
  $('#object-note').textContent=rectangular?(type==='pedestrian'?'生成一个背向自然行走的行人。':'生成一个简单感叹号警示牌，不依赖文字。'):dustbin?'生成一个直立垃圾桶，保留区域外的场景。':'沿检测到的车道线添加交通锥，保留原有生成方式。';
  $('#start-note').textContent=rectangular?'先点击预览选位置，再启动生成。生成后请审核主体与场景。':'自动生成遮罩后直接生图，无需手动导入工作流。';
  $('#placement-note').textContent=manual?'点击图片设置底部接地点。':'自动位置仅依据车道线，未检测现有障碍物；请核对生成结果，必要时改用手动选位。';
  drawPlacement();buttonState();
}
let previousType=$('#object-type').value;
$('#object-type').onchange=()=>{
  prompts[previousType]=$('#prompt').value;placementDrafts[previousType]=placementValues(previousType);
  previousType=$('#object-type').value;placementChosen=false;ready=false;
  if(rectangleType())$('#placement-mode').value='manual';
  pendingDefaults=applyPlacementDefaults(previousType)?null:previousType;
  placementControls();checkHealth(true);
};
$('#placement-mode').onchange=()=>{ready=false;placementControls();checkHealth();};
$('#protect-original').onchange=()=>{ready=false;checkHealth();};
$('#generation-preset').onchange=()=>{const value=$('#generation-preset').value;if(value!=='custom')$('#steps').value=value;else $('.advanced').open=true;};
$('#steps').oninput=()=>{$('#generation-preset').value=['20','30'].includes($('#steps').value)?$('#steps').value:'custom';};
for(const id of ['place-x','place-y','place-width','place-height'])$('#'+id).oninput=()=>{placementChosen=true;pendingDefaults=null;drawPlacement();buttonState();};
$('#placement-canvas').onpointerdown=e=>{
  const r=e.currentTarget.getBoundingClientRect();
  $('#place-x').value=(Math.max(.05,Math.min(.95,(e.clientX-r.left)/r.width))*100).toFixed(1);
  $('#place-y').value=(Math.max(.25,Math.min(.96,(e.clientY-r.top)/r.height))*100).toFixed(1);
  placementChosen=true;pendingDefaults=null;
  drawPlacement();buttonState();
};
function reviewPoint(e){const r=$('#review-canvas').getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))];}
function rectangle(a,b){return [Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.abs(a[0]-b[0]),Math.abs(a[1]-b[1])];}
function drawReview(){
  if(!reviewImage?.naturalWidth)return;
  const c=$('#review-canvas');c.width=Math.min(1024,reviewImage.naturalWidth);c.height=Math.round(c.width*reviewImage.naturalHeight/reviewImage.naturalWidth);
  const ctx=c.getContext('2d');ctx.drawImage(reviewImage,0,0,c.width,c.height);ctx.strokeStyle='#d2f56b';ctx.lineWidth=2;
  const boxes=dragStart&&dragEnd?[...reviewBoxes,rectangle(dragStart,dragEnd)]:reviewBoxes;
  boxes.forEach(([x,y,w,h],i)=>{ctx.strokeRect(x*c.width,y*c.height,w*c.width,h*c.height);ctx.fillStyle='#d2f56b';ctx.fillText(String(i+1),x*c.width+4,y*c.height+14);});
}
function loadReview(job){
  $('#review-panel').hidden=job.status!=='done';
  $('#review-state').textContent={accepted:'已通过',rejected:'不合格'}[job.review?.verdict]||'待审核';
  if(job.review?.reviewer==='codex_visual_review')$('#review-state').textContent+=' · Codex 辅助审阅';
  if(job.status!=='done'||reviewJobId===job.id)return;
  reviewJobId=job.id;reviewBoxes=job.review?.boxes?.map(b=>[...b])||[];dragStart=dragEnd=null;
  $('#review-notes').value=job.review?.notes||'';reviewImage=new Image();reviewImage.onload=drawReview;reviewImage.src=job.files.result;
}
$('#review-canvas').onpointerdown=e=>{e.preventDefault();dragStart=reviewPoint(e);dragEnd=dragStart;e.currentTarget.setPointerCapture(e.pointerId);};
$('#review-canvas').onpointermove=e=>{if(dragStart){dragEnd=reviewPoint(e);drawReview();}};
$('#review-canvas').onpointerup=e=>{if(!dragStart)return;const box=rectangle(dragStart,reviewPoint(e));if(box[2]>.005&&box[3]>.005)reviewBoxes.push(box);dragStart=dragEnd=null;drawReview();};
$('#review-canvas').onpointercancel=()=>{dragStart=dragEnd=null;drawReview();};
$('#clear-boxes').onclick=()=>{reviewBoxes=[];drawReview();};
async function saveReview(verdict){
  const id=current?.id;if(!id)return;
  try{error();const job=await api('api/jobs/'+id+'/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({verdict,boxes:reviewBoxes,notes:$('#review-notes').value})});if(current?.id===id)showJob(job);}
  catch(e){error(e.message);}
}
$('#accept-review').onclick=()=>saveReview('accepted');$('#reject-review').onclick=()=>saveReview('rejected');
placementControls();checkHealth();loadHistory(true);setInterval(poll,1500);setInterval(()=>checkHealth(),20000);
