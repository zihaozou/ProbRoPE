// Retry media within the browser's user-activation event, without timers.
window.paperMedia = (() => {
  const pending = new WeakSet();
  function configure(video) {
    video.muted = true;
    video.defaultMuted = true;
    video.playsInline = true;
    for (const name of ['muted', 'playsinline', 'webkit-playsinline', 'x5-playsinline']) video.setAttribute(name, '');
  }
  function play(video) {
    configure(video);
    if (!video.paused || pending.has(video)) return;
    pending.add(video);
    Promise.resolve(video.play()).then(() => {
      delete video.dataset.playBlocked;
    }).catch(error => {
      if (error.name === 'NotAllowedError') video.dataset.playBlocked = 'true';
    }).finally(() => pending.delete(video));
  }
  const retry = () => document.dispatchEvent(new Event('paper-media-activation'));
  for (const name of ['touchend', 'click', 'keydown', 'WeixinJSBridgeReady']) document.addEventListener(name, retry);
  window.addEventListener('pageshow', retry);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) retry(); });
  return {configure, play};
})();

/* Native PPT entrance effects rebuilt as per-element GSAP timelines. */
(() => {
  if (!window.gsap) return;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  // Reparent the original animated elements for narrow screens: no duplicate
  // IDs, duplicate media decoders, or flattened slide images.
  const narrow = matchMedia('(max-width: 760px), (max-width: 900px) and (orientation: portrait)');
  const layouts = [...document.querySelectorAll('.animation-page:not(.pipeline-page) .slide-stage')].map(stage => {
    const source=stage.querySelector('.slide-art');
    const groups=[...source.children].filter(g=>g.dataset.shape);
    const title=groups[0];
    const mobile=document.createElement('div'); mobile.className='slide-mobile';
    const heading=document.createElement('h2');
    const titleLines=[...title.querySelectorAll('foreignObject > div > div')];
    titleLines.forEach((line,index)=>{
      if(index)heading.append(document.createTextNode(' '));
      for(const run of line.children){
        const span=document.createElement('span');span.textContent=run.textContent;
        // Preserve semantic color emphasis without carrying PPT font sizes.
        if(run.style.color)span.style.color=run.style.color;
        heading.append(span);
      }
    });
    if(!heading.textContent)heading.textContent=title.textContent;
    heading.className='page-heading slide-heading';stage.closest('[data-page]').prepend(heading);title.style.display='none';
    source.setAttribute('viewBox','0 140 1280 580');
    const makeSVG=(viewBox,label) => {
      const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
      svg.setAttribute('viewBox',viewBox); svg.setAttribute('role','img'); svg.setAttribute('aria-label',label);
      svg.classList.add('mobile-art'); mobile.append(svg); return svg;
    };
    const slide=Number(stage.closest('[data-slide]').dataset.slide);
    const diagram=makeSVG(slide===1?'25 150 590 520':slide===2?'20 150 390 550':'20 245 390 310','Temporal encoding diagram');
    let secondDiagram=null;
    if(slide===2){
      diagram.setAttribute('viewBox','20 157 390 250');
      secondDiagram=makeSVG('20 410 390 280','Physical timestamp encoding');
      const pair=document.createElement('div');pair.className='mobile-diagram-pair';
      mobile.prepend(pair);pair.append(diagram,secondDiagram);
    }
    let output;
    if(slide===1) output=makeSVG('580 155 715 465','Patchifier and token sequences');
    else {
      const comparison=document.createElement('div'); comparison.className='slide-media';
      comparison.setAttribute('aria-label','Video comparison');
      output=makeSVG('448 260 792 295','Low FPS input, ground truth, and reconstruction');
      comparison.append(output); mobile.append(comparison);
    }
    // Keep video in ordinary HTML: WebKit composites video outside SVG
    // foreignObject coordinates and ignores the surrounding SVG clip path.
    const video=source.querySelector('video');
    let media=null;
    if(video){
      media=document.createElement('div');media.className='slide-comparison';
      const labels=document.createElement('div');labels.className='comparison-labels';
      ['Low FPS input','GT',slide===2?'RoPE':'ProbRoPE'].forEach((text,i)=>{
        const label=document.createElement('span');label.textContent=text;
        label.style.color=['#2378a3','#555',slide===2?'#CC8178':'#A02B93'][i];labels.append(label);
      });
      const crop=document.createElement('div');crop.className='comparison-crop';
      video.removeAttribute('style');crop.append(video);
      for(let i=0;i<2;i++){const gap=document.createElement('span');gap.className='comparison-gutter';gap.style.left=(i?66.09589:32.19178)+'%';crop.append(gap);}
      media.append(labels,crop);stage.append(media);
      const original=source.querySelector('[data-shape="700"]');original.replaceChildren();
      media.dataset.animationTarget=original.id;
    }
    stage.append(mobile);
    const outputIds=new Set([145,80,146,151,156,161,166,171,95,96,600]);
    return {stage,source,groups,title,mobile,diagram,secondDiagram,output,slide,outputIds,media};
  });
  function layoutSlides() {
    for(const l of layouts) {
      if(narrow.matches) {
        for(const group of l.groups) {
          if(group===l.title) continue;
          const isOutput=l.slide===1?l.outputIds.has(Number(group.dataset.shape)):group.dataset.shape==='700';
          const shape=Number(group.dataset.shape);
          const second=l.slide===2 && ([601,621,631].includes(shape)||shape>=2072);
          (isOutput?l.output:second?l.secondDiagram:l.diagram).append(group);
        }
        if(l.media){l.output.style.display='none';l.output.parentElement.append(l.media);}
      } else {
        l.groups.forEach(g=>l.source.append(g));
        if(l.media)l.stage.append(l.media);
      }
    }
    fitSlides();
  }
  function fitSlides() {
    for (const l of layouts) {
      if (!narrow.matches) {
        l.stage.style.removeProperty('height');
        l.mobile.style.removeProperty('transform');
        continue;
      }
      const page=l.stage.closest('[data-page]');
      const css=getComputedStyle(page);
      const controls=page.querySelector('.slide-controls');
      const available=Math.max(1,page.clientHeight-parseFloat(css.paddingTop)-parseFloat(css.paddingBottom));
      if(l.media){
        const diagramSpace=Math.max(1,available-l.media.offsetHeight-18);
        if(l.secondDiagram){
          for(const diagram of [l.diagram,l.secondDiagram])diagram.style.maxHeight=`${diagramSpace}px`;
        }else l.diagram.style.maxHeight=`${diagramSpace}px`;
      }
      const natural=l.mobile.offsetHeight;
      const scale=Math.min(1,available/Math.max(1,natural));
      l.mobile.style.transform=`scale(${scale})`;
      l.stage.style.height=`${natural*scale}px`;
    }
    document.dispatchEvent(new Event('paper-layout'));
  }
  const sizeObserver=new ResizeObserver(fitSlides);
  layouts.forEach(l=>{sizeObserver.observe(l.stage.closest('[data-page]'));sizeObserver.observe(l.mobile);sizeObserver.observe(l.stage.closest('[data-page]').querySelector('.slide-controls'));});
  document.fonts.ready.then(fitSlides);
  layoutSlides();
  narrow.addEventListener('change',layoutSlides);
  const entries = new Map();
  let active = null;
  let mediaTarget = null;
  for (const spec of window.slideTimingData || []) {
    const section = document.querySelector(`[data-slide="${spec.slide}"]`);
    const timeline = gsap.timeline({paused:true});
    const animated = new Set();
    for (const effect of spec.animations) {
      const target = section.querySelector(`[data-animation-target="${effect.target.replace(/^#/,'')}"]`) || section.querySelector(effect.target);
      if (!target) continue;
      animated.add(target);
      const from = {opacity:0};
      const to = {opacity:1,duration:effect.duration,ease:'none'};
      if (effect.filter==='pdf-element') {
        from.x=-2.5; to.x=0; to.ease='power2.out';
      }
      if (effect.filter==='pdf-wire') {
        from.clipPath='inset(0 100% 0 0)';
        to.clipPath='inset(0 0 0 0)';
        to.ease='power1.inOut';
      }
      if (effect.filter==='pipeline') {
        to.ease='power2.out';
        const wire=target.dataset.name?.includes('Connector');
        if (wire) {
          for (const path of target.querySelectorAll('path')) {
            const length=path.getTotalLength();
            timeline.fromTo(path,{strokeDasharray:length,strokeDashoffset:length},{strokeDashoffset:0,duration:effect.duration,ease:'none'},effect.start);
          }
        } else if (!target.querySelector('foreignObject')) {
          from.x=-9; to.x=0;
        }
      }
      if (effect.filter.startsWith('wipe')) {
        from.clipPath = effect.filter.includes('right') ? 'inset(0 0 0 100%)' : 'inset(0 100% 0 0)';
        to.clipPath = 'inset(0 0 0 0)';
      }
      if (effect.scale !== undefined) {
        from.scale = effect.scale;
        to.scale = 1;
        gsap.set(target,{transformOrigin:'50% 50%'});
      }
      if (effect.motion) {
        const m = effect.motion;
        from[m.axis] = (m.from-m.to)*(m.axis==='y'?720:1280);
        to[m.axis] = 0;
      }
      timeline.fromTo(target,from,to,effect.start);
    }
    // Each comparison retains the single synchronized video from the PPT.
    const videos = [...section.querySelectorAll('video')];
    videos.forEach(video => { window.paperMedia.configure(video); video.preload='auto'; video.addEventListener('error',()=>{section.dataset.mediaError='true';}); });
    const pauseButton = section.querySelector('[data-pause]');
    const entry = {section,timeline,videos,animated,playing:false};
    entries.set([...document.querySelectorAll('[data-page]')].indexOf(section),entry);
    const start = (restartMedia=true) => {
      entry.playing=true;
      pauseButton.textContent='Pause';
      if (reduced.matches) timeline.progress(1).pause();
      else timeline.restart();
      videos.forEach(v=>{if(restartMedia)v.currentTime=0;if(!reduced.matches)window.paperMedia.play(v);});
    };
    section.querySelector('[data-replay]').addEventListener('click',()=>start(true));
    pauseButton.addEventListener('click',()=>{
      entry.playing=!entry.playing;
      pauseButton.textContent=entry.playing?'Pause':'Play';
      if(entry.playing){timeline.resume();videos.forEach(v=>window.paperMedia.play(v));}
      else{timeline.pause();videos.forEach(v=>v.pause());}
    });
    entry.start=start;
    if (reduced.matches) timeline.progress(1).pause();
  }
  // Start the destination's media on navigation intent, before the page moves.
  // Diagram timelines remain separate; arrival must not rewind an already playing video.
  window.preparePaperSlide = index => {
    const next=entries.get(index);
    if(mediaTarget===next) return;
    mediaTarget?.videos.forEach(video=>video.pause());
    mediaTarget=next;
    if(!next) return;
    next.videos.forEach(video=>{
      video.currentTime=0;
      if(!reduced.matches) window.paperMedia.play(video);
    });
  };
  window.activatePaperSlide = index => {
    const next = entries.get(index);
    if (active===next) return;
    if(active) active.timeline.pause();
    active=next;
    if(active) active.start(false);
  };
  document.addEventListener('visibilitychange',()=>{
    if(document.hidden){
      mediaTarget?.videos.forEach(v=>v.pause());
      if(active)active.timeline.pause();
    }
  });
  document.addEventListener('paper-media-activation',()=>{
    if(document.hidden || reduced.matches)return;
    if(mediaTarget && (mediaTarget!==active || active.playing))mediaTarget.videos.forEach(v=>window.paperMedia.play(v));
    if(active?.playing)active.timeline.resume();
  });
  reduced.addEventListener('change',()=>{if(reduced.matches)entries.forEach(e=>{e.timeline.progress(1).pause();e.videos.forEach(v=>v.pause());});});
})();
