/* Epoch-40 x8. The same RGB/event inputs and target cameras are used for
 * every method. Source paths and encoding details are recorded in WEBSITE.md.
 */
window.resultComparisonData = {
  temporalScale:8,
  cameras:[3,7,11],
  methods: [
    {id:'dfuse',label:'D-FUSE'},
    {id:'movies',label:'MoVieS'}, {id:'gcd',label:'GCD'},
    {id:'tc',label:'TrajectoryCrafter'}
  ],
  scenes: [2900,2910,2920,2940,2960,2980].map(number=>{
    const id=`scn${String(number).padStart(5,'0')}`;
    return {id,label:`Scene ${number}`,rgb:`assets/result-${id}-rgb.mp4`,
      event:`assets/result-${id}-event.mp4`,
      methods:Object.fromEntries(['dfuse','movies','gcd','tc'].map(method=>[
        method,[3,7,11].map(camera=>`assets/result-${id}-${method}-cam${camera}.mp4`)
      ]))};
  })
};

// Fetch once with bounded concurrency. Blob URLs are reused across method switches,
// including browsers that do not reuse fetch responses for video range requests.
const resultMediaCache=(()=>{
  const entries=new Map(),queue=[];
  let active=0;
  function pump(){
    queue.sort((a,b)=>b.priority-a.priority);
    while(active<4 && queue.length){
      const job=queue.shift();job.started=true;active++;
      fetch(job.src,{cache:'force-cache',priority:job.priority>1?'high':'low'})
        .then(response=>{if(!response.ok)throw new Error(`Video ${response.status}`);return response.blob();})
        .then(blob=>{job.url=URL.createObjectURL(blob);job.resolve(job.url);})
        .catch(error=>{entries.delete(job.src);job.reject(error);})
        .finally(()=>{active--;pump();});
    }
  }
  function request(src,priority=0){
    if(!src)return Promise.resolve(null);
    if(/MicroMessenger/i.test(navigator.userAgent))return Promise.resolve(src);
    let job=entries.get(src);
    if(job){job.priority=Math.max(job.priority,priority);pump();return job.promise;}
    job={src,priority,started:false};
    job.promise=new Promise((resolve,reject)=>{job.resolve=resolve;job.reject=reject;});
    entries.set(src,job);queue.push(job);pump();return job.promise;
  }
  return {request,prioritize(src){request(src,3).catch(()=>{});}};
})();

// Starting every decoder in one task stalls WebKit's media pipeline. File
// prefetching is independent; attach at most two live players per render frame.
const resultDecoderQueue=[];
let resultDecoderFrame=0;
function queueResultDecoder(attach){
  resultDecoderQueue.push(attach);
  if(resultDecoderFrame)return;
  function pump(){
    resultDecoderFrame=0;
    for(let count=0;count<2 && resultDecoderQueue.length;count++)resultDecoderQueue.shift()();
    if(resultDecoderQueue.length)resultDecoderFrame=requestAnimationFrame(pump);
  }
  resultDecoderFrame=requestAnimationFrame(pump);
}

function initializeResultGallery(page,dataset,prefix) {
  if(!page || !dataset) return;
  const {methods,scenes}=dataset;
  const tabs=page.querySelector('[role="tablist"]');
  const panel=page.querySelector('[role="tabpanel"]');
  const gallery=page.querySelector('.results-gallery');
  const playButton=page.querySelector('[data-results-play]');
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  let selected=methods[0].id, pageVisible=false, pageNear=false, playing=!reduced.matches, playbackFrame=0;
  const visibleRows=new Set(), rows=[];
  const cellMedia=new WeakMap();
  let initialized=false;
  const columnNames=['RGB input','Event input','Novel view 1','Novel view 2','Novel view 3'];
  function syncPlayback(row) {
    const videos=[...row.querySelectorAll('video')];
    const shouldPlay=pageVisible && visibleRows.has(row) && playing && !document.hidden;
    videos.forEach(video=>{
      if(pageVisible && visibleRows.has(row) && !video.hidden) resultMediaCache.prioritize(video.dataset.src);
      if(shouldPlay && !video.hidden && video.getAttribute('src')) {
        if(!video.paused)return;
        window.paperMedia.play(video,()=>video.isConnected&&!document.hidden&&pageVisible&&visibleRows.has(row)&&playing&&!video.hidden);
      } else if(!video.paused)video.pause();
    });
  }
  function residentRows(){
    if(document.hidden)return new Set();
    if(pageVisible)return visibleRows;
    // One predecoded row on approach; all other media are already prefetched.
    return new Set(pageNear?rows.slice(0,1):[]);
  }
  function refreshVisibleRows(){
    const bounds=gallery.getBoundingClientRect();
    const viewportHeight=window.visualViewport?.height || window.innerHeight;
    const top=Math.max(0,bounds.top),bottom=Math.min(viewportHeight,bounds.bottom);
    pageVisible=bottom>top && bounds.right>0 && bounds.left<window.innerWidth;
    visibleRows.clear();
    if(!pageVisible)return;
    rows.forEach(row=>{
      const rect=row.getBoundingClientRect();
      if(Math.min(rect.bottom,bottom)-Math.max(rect.top,top)>1)visibleRows.add(row);
    });
  }
  function updatePlayback() {
    refreshVisibleRows();
    const blocked=rows.some(row=>visibleRows.has(row)&&row.querySelector('video[data-play-blocked]'));
    playButton.textContent=playing&&!blocked?'Pause videos':'Play videos';
    const resident=residentRows();
    rows.forEach(row=>{if(resident.has(row)){hydrateRow(row);syncPlayback(row);}else releaseRow(row);});
  }
  function schedulePlayback(){
    if(playbackFrame)return;
    playbackFrame=requestAnimationFrame(()=>{playbackFrame=0;updatePlayback();});
  }
  function fillCell(cell,src,label) {
    let state=cellMedia.get(cell);
    if(!state){state={videos:new Map(),requested:null,shown:null};cellMedia.set(cell,state);}
    if(state.requested===src)return;
    state.requested=src;cell.setAttribute('aria-busy','true');
    // Rapid switches must not leave superseded pending decoders alive.
    for(const [otherSrc,other] of state.videos){if(otherSrc!==src && otherSrc!==state.shown){disposeVideo(other);state.videos.delete(otherSrc);}}
    if(!src){state.videos.forEach(v=>{v.pause();v.hidden=true;});state.shown=null;cell.removeAttribute('aria-busy');return;}
    const row=cell.closest('.result-row');
    function present(video){
      if(state.requested!==src || !video.isConnected || video.readyState<2 || video.seeking)return;
      state.videos.forEach(other=>{other.hidden=other!==video;if(other!==video)other.pause();});
      state.shown=src;
      // Keep the old frame until this one is ready, then release its decoder.
      for(const [otherSrc,other] of state.videos){if(other!==video){disposeVideo(other);state.videos.delete(otherSrc);}}
      cell.querySelector('.result-missing')?.remove();
      cell.classList.add('media-ready');cell.removeAttribute('aria-busy');syncPlayback(row);
    }
    function prepare(video){
      if(state.requested!==src || video.readyState<2)return;
      const reference=row.querySelector('.result-cell:first-child video');
      if(reference && reference!==video && Number.isFinite(video.duration)){
        const time=reference.currentTime%video.duration;
        if(Math.abs(video.currentTime-time)>.1){video.currentTime=time;return;}
      }
      present(video);
    }
    let video=state.videos.get(src);
    if(video){video.onseeked=()=>present(video);prepare(video);return;}
    video=document.createElement('video');video.hidden=Boolean(state.shown);
    window.paperMedia.configure(video);video.loop=true;video.preload='auto';
    video.defaultPlaybackRate=dataset.playbackRate || 1;video.playbackRate=dataset.playbackRate || 1;
    video.dataset.src=src;video.setAttribute('aria-label',label);
    state.videos.set(src,video);cell.append(video);
    video.addEventListener('loadedmetadata',()=>{video.playbackRate=dataset.playbackRate || 1;syncPlayback(row);});
    video.addEventListener('loadeddata',()=>prepare(video));
    video.onseeked=()=>present(video);
    video.addEventListener('error',()=>{
      if(state.requested!==src)return;
      state.videos.forEach(v=>{v.pause();v.hidden=true;});cell.removeAttribute('aria-busy');
      cell.querySelector('.result-missing')?.remove();
      const message=document.createElement('span');message.className='result-missing';message.textContent='Video unavailable';cell.append(message);
    });
    const attach=source=>queueResultDecoder(()=>{if(video.isConnected && state.requested===src){video.src=source;video.load();syncPlayback(row);}});
    resultMediaCache.request(src,pageVisible&&visibleRows.has(row)?3:1).then(attach).catch(()=>attach(src));
  }
  function hydrateRow(row){
    const scene=scenes[rows.indexOf(row)],cells=row.querySelectorAll('.result-cell');
    fillCell(cells[0],scene.rgb,`${scene.label}, RGB input`);
    fillCell(cells[1],scene.event,`${scene.label}, Event input`);
    hydrateMethod(row);
  }
  function hydrateMethod(row){
    const index=rows.indexOf(row),scene=scenes[index],method=methods.find(m=>m.id===selected);
    const views=scene.methods[selected]||[],cells=row.querySelectorAll('.result-cell');
    for(let i=0;i<3;i++)fillCell(cells[i+2],views[i],`${scene.label}, ${method.label}, novel view ${i+1}`);
  }
  function disposeVideo(video){
    video.pause();video.onseeked=null;video.removeAttribute('src');video.load();video.remove();
  }
  function releaseRow(row){
    row.querySelectorAll('.result-cell').forEach(cell=>{
      const state=cellMedia.get(cell);if(!state || !state.videos.size)return;
      state.requested=null;state.shown=null;
      state.videos.forEach(disposeVideo);state.videos.clear();
      cell.classList.remove('media-ready');cell.removeAttribute('aria-busy');
    });
  }
  scenes.forEach(scene=>{
    const row=document.createElement('article');row.className='result-row';row.dataset.scene=scene.id;
    row.setAttribute('aria-label',scene.label);
    const grid=document.createElement('div');grid.className='result-grid';row.append(grid);
    columnNames.forEach((name,i)=>{
      const cell=document.createElement('div');cell.className='result-cell';cell.setAttribute('aria-label',`${scene.label}, ${name}`);
      grid.append(cell);
    });
    gallery.append(row);rows.push(row);
  });
  function chooseMethod(id) {
    if(initialized&&selected===id)return;
    initialized=true;selected=id;
    const method=methods.find(m=>m.id===id);
    [...tabs.children].forEach(button=>{
      const active=button.dataset.method===id;
      button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;
    });
    panel.setAttribute('aria-labelledby',`${prefix}-method-${id}`);
    // One update per frame, including rapid method changes and row observers.
    schedulePlayback();
    const hasVideos=scenes.length>0;
    playButton.disabled=!hasVideos;
    page.querySelector('.results-footer').hidden=!hasVideos;
  }
  methods.forEach(method=>{
    const button=document.createElement('button');button.type='button';button.role='tab';
    button.id=`${prefix}-method-${method.id}`;button.dataset.method=method.id;
    button.setAttribute('aria-controls',panel.id);button.textContent=method.label;
    button.addEventListener('click',()=>chooseMethod(method.id));tabs.append(button);
  });
  tabs.addEventListener('keydown',event=>{
    const index=methods.findIndex(m=>m.id===selected);
    let next;
    if(event.key==='ArrowRight')next=(index+1)%methods.length;
    if(event.key==='ArrowLeft')next=(index+methods.length-1)%methods.length;
    if(event.key==='Home')next=0;
    if(event.key==='End')next=methods.length-1;
    if(next===undefined)return;
    event.preventDefault();chooseMethod(methods[next].id);tabs.children[next].focus();
  });
  // Native scrolling chains to page navigation at either gallery boundary.
  const rowObserver=new IntersectionObserver(entries=>{
    entries.forEach(({target,isIntersecting})=>{
      if(isIntersecting)visibleRows.add(target);else visibleRows.delete(target);
    });
    schedulePlayback();
  },{root:gallery,threshold:0.05});
  rows.forEach(row=>rowObserver.observe(row));
  gallery.addEventListener('scroll',schedulePlayback,{passive:true});
  window.addEventListener('scroll',schedulePlayback,{passive:true});
  window.addEventListener('resize',schedulePlayback,{passive:true});
  window.visualViewport?.addEventListener('resize',schedulePlayback,{passive:true});
  new IntersectionObserver(entries=>{
    pageVisible=entries[0].intersectionRatio>=0.5;schedulePlayback();
  },{threshold:0.5}).observe(page);
  // Network data are warm before arrival, without keeping an entire hidden
  // gallery's decoders allocated beside the visible gallery.
  new IntersectionObserver(entries=>{
    pageNear=entries[0].isIntersecting;schedulePlayback();
  },{rootMargin:'100% 0px',threshold:0}).observe(page);
  playButton.addEventListener('click',()=>{
    const blocked=rows.some(row=>visibleRows.has(row)&&row.querySelector('video[data-play-blocked]'));
    playing=blocked?true:!playing;updatePlayback();
  });
  document.addEventListener('paper-media-activation',updatePlayback);
  document.addEventListener('visibilitychange',updatePlayback);
  reduced.addEventListener('change',()=>{playing=!reduced.matches;updatePlayback();});
  // Fit complete rows to the available gallery area.
  function fitGallery() {
    const header=panel.querySelector('.result-columns');
    const available=Math.max(1,panel.clientHeight-header.offsetHeight-10);
    const gap=10;
    const measuredWidth=dataset.videoAspectRatio ? panel.clientWidth-16 : gallery.clientWidth;
    const natural=(measuredWidth-40)/5/(dataset.videoAspectRatio || 1.5);
    const minimumRowHeight=Math.min(natural,120);
    const maxRows=matchMedia('(max-width:760px) and (orientation:portrait)').matches?6:3;
    const count=Math.min(rows.length,maxRows,Math.max(1,Math.floor((available+gap)/(minimumRowHeight+gap))));
    const rowHeight=Math.max(1,Math.min(natural,(available-gap*(count-1))/count));
    if(dataset.videoAspectRatio) {
      const videoSize=Math.max(1,Math.min(rowHeight,(panel.clientWidth-48)/5));
      panel.style.setProperty('--realworld-video-size',`${videoSize}px`);
      page.querySelector('.results-footer').style.width=`${Math.min(panel.clientWidth,videoSize*5+48)}px`;
    }
    gallery.style.setProperty('--result-row-height',`${rowHeight}px`);
    gallery.style.height=`${rowHeight*count+gap*Math.max(0,count-1)}px`;
    schedulePlayback();
  }
  new ResizeObserver(fitGallery).observe(panel);
  document.fonts.ready.then(fitGallery);
  chooseMethod(selected);updatePlayback();fitGallery();
}
initializeResultGallery(document.querySelector('[aria-labelledby="results-title"]'),window.resultComparisonData,'result');
initializeResultGallery(document.querySelector('#real-world-results'),window.realWorldComparisonData,'realworld-result');

// Enqueue alternates after both galleries have requested their default videos.
for(const dataset of [window.resultComparisonData,window.realWorldComparisonData]){
  if(!dataset)continue;
  for(const scene of dataset.scenes){
    for(const sources of [[scene.rgb,scene.event],...Object.values(scene.methods)]){
      for(const src of sources)resultMediaCache.request(src,0).catch(()=>{});
    }
  }
}
