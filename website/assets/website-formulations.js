/* Horizontally sliding derivation chapters with a persistent timeline. */
(()=>{
  const page=document.querySelector('#probrope-formulations');if(!page)return;
  const reduced=matchMedia('(prefers-reduced-motion: reduce)'),content=page.querySelector('.formulation-content'),viewport=page.querySelector('.formulation-viewport');
  const title=page.querySelector('h2'),heading=page.querySelector('.formulation-heading'),cardsElement=page.querySelector('.formulation-cards');
  const storyTemplates=[...page.querySelectorAll('[data-story-equation]')];
  const formsChapter=storyTemplates.length-1;
  const mobileForms=matchMedia('(max-width:760px)');
  const chapterCount=()=>storyTemplates.length+(mobileForms.matches?2:0);
  const chapterTimes=()=>mobileForms.matches?[0,4,13,18,24,30]:[0,4,13,18];
  const pauseButton=page.querySelector('[data-formula-pause]');
  let master=null,chapter=0,entryChapter=0,manualEntry=false,active=false,paused=false,resizeTimer;
  const effects=new Map();
  function stage(element){const source=document.createElement('div');source.className='formula-source';element.append(source);return {element,source,index:0};}
  const story=stage(page.querySelector('.formula-story'));
  const cards=[...page.querySelectorAll('.formulation-card')].map(el=>({el}));
  const svgNS='http://www.w3.org/2000/svg';
  const temporal=document.createElement('div');temporal.className='formula-temporal';
  story.element.before(temporal);
  const originalToken=document.querySelector('[data-slide="3"] path[fill="#92D07B"]');
  function node(tag,attrs={}){const el=document.createElementNS(svgNS,tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);return el;}
  function label(svg,text,x,y,size=19){const el=node('text',{x,y,'text-anchor':'middle','font-size':size,'font-family':"Cambria, 'Times New Roman', serif",fill:'#202020'});el.textContent=text;svg.append(el);return el;}
  function timeline(kind,id){
    const svg=node('svg',{viewBox:'0 0 600 210',role:'img','aria-label':kind==='story'?'A token on a timeline: a timestamp becomes a temporal distribution':`${kind} temporal distribution illustration`});
    const defs=node('defs'),gradient=node('linearGradient',{id,x1:0,y1:0,x2:0,y2:1});
    gradient.append(node('stop',{offset:0,'stop-color':'#92D07B','stop-opacity':.9}),node('stop',{offset:1,'stop-color':'#92D07B','stop-opacity':.12}));defs.append(gradient);svg.append(defs);
    const token=node('svg',{x:190,y:4,width:220,height:60,viewBox:'0 482 44 23',preserveAspectRatio:'none'});
    if(originalToken){token.append(originalToken.cloneNode(true));if(originalToken.nextElementSibling?.tagName.toLowerCase()==='path')token.append(originalToken.nextElementSibling.cloneNode(true));}
    svg.append(token);
    const distribution=node('g',{'data-temporal-distribution':''});
    if(kind==='story'){
      distribution.append(node('image',{href:'assets/ppt-alias-image30.svg',x:190,y:71,width:220,height:82,preserveAspectRatio:'none'}));
    }else{
      const samples=Array.from({length:121},(_,i)=>{const z=i/120;let h=0;
        if(kind==='Uniform')h=1;
        else if(kind==='Gaussian')h=Math.exp(-.5*((z-.5)/.17)**2);
        else h=.82*Math.exp(-.5*((z-.2)/.065)**2)+Math.exp(-.5*((z-.51)/.085)**2)+.7*Math.exp(-.5*((z-.8)/.06)**2);
        return [190+220*z,153-76*h];});
      const curve=samples.map(([x,y],i)=>`${i?'L':'M'}${x.toFixed(2)} ${y.toFixed(2)}`).join(' ');
      distribution.append(node('path',{d:`M190 153 ${curve.replace(/^M/,'L')} L410 153 Z`,fill:`url(#${id})`,stroke:'none'}));
      distribution.append(node('path',{d:kind==='Uniform'?`M190 153 L190 77 L410 77 L410 153`:curve,fill:'none',stroke:'#202020','stroke-width':1.6,'stroke-linejoin':'round'}));
    }
    svg.append(distribution);
    svg.append(node('path',{d:'M60 155 Q280 153.8 541 155 M532 148 L542 155 L532 162',fill:'none',stroke:'#202020','stroke-width':1.7,'stroke-linecap':'round','stroke-linejoin':'round'}));
    for(let x=100;x<=500;x+=50)svg.append(node('path',{d:`M${x} 150 L${x+.3} 160`,stroke:'#202020','stroke-width':1.2}));
    svg.append(node('path',{d:'M190 62 V158 M410 62 V158',stroke:'#777','stroke-width':1,'stroke-dasharray':'4 5',fill:'none'}));
    const center=node('path',{d:'M300 64 V157',stroke:'#202020','stroke-width':1.3,'stroke-dasharray':'5 5',fill:'none'});svg.append(center);
    label(svg,'t',561,163,23);label(svg,'t − Δt/2',190,199,17);label(svg,'t + Δt/2',410,199,17);
    const marker=node('g',{'data-temporal-timestamp':''});marker.append(node('image',{href:'assets/ppt-alias-image26.svg',x:270,y:147,width:60,height:44,preserveAspectRatio:'none'}));label(marker,'t',300,181,20);svg.append(marker);
    if(kind!=='story'){marker.setAttribute('opacity','0');label(svg,'t',300,179,19);}
    return {svg,distribution,marker};
  }
  const timeDiagram=timeline('story','formula-time-fill');temporal.append(timeDiagram.svg);
  cards.forEach((card,i)=>{
    const plot=document.createElement('div');plot.className='formula-distribution';
    const name=card.el.querySelector('h3').textContent;const diagram=timeline(name,`formula-distribution-${i}`);plot.append(diagram.svg);card.el.querySelector('.formula-answer').before(plot);card.plot=plot;card.distribution=diagram.distribution;
    if(name==='Learned'){const note=document.createElement('span');note.className='distribution-caption';note.textContent='Illustrative learned shape';plot.append(note);}
  });
  let temporalMode=null;
  function updateTemporal(index,animate){
    const mode=index===0?'timestamp':index>=formsChapter?'forms':'distribution';
    temporal.hidden=mode==='forms';
    if(mode===temporalMode)return; // Keep the same distribution through substitution and factorization.
    temporalMode=mode;
    cancelEffect(temporal);
    if(mode==='forms'){
      if(animate&&!reduced.matches){const tween=gsap.fromTo(cards.map(c=>c.distribution),{scaleY:0,opacity:0,svgOrigin:'300 155'},{scaleY:1,opacity:1,duration:.65,stagger:.08,ease:'power2.out'});effects.set(temporal,tween);}
      return;
    }
    const duration=animate&&!reduced.matches?.65:0;
    gsap.set(timeDiagram.distribution,{svgOrigin:'300 153'});
    const tween=gsap.timeline();effects.set(temporal,tween);
    tween.to(timeDiagram.distribution,{scaleX:mode==='distribution'?1:0,opacity:mode==='distribution'?1:0,duration,ease:'power2.inOut'},0)
      .to(timeDiagram.marker,{opacity:mode==='timestamp'?1:0,duration:duration*.7},0);
  }
  function cancelEffect(box){effects.get(box)?.kill();effects.delete(box);box.querySelectorAll('.formula-ghost').forEach(el=>el.remove());const source=box.querySelector(':scope>.formula-source');if(source)source.style.opacity='1';if(box===heading)gsap.set(title,{clearProps:'opacity,transform'});}
  function sizeEquation(s){const math=s.source.querySelector('math');if(!math)return;math.style.transform='none';const scale=s.element.getBoundingClientRect().width/s.element.clientWidth||1;const natural=math.getBoundingClientRect().width/scale;math.style.transform=`scale(${Math.min(1,(s.element.clientWidth-12)/Math.max(1,natural))})`;}
  function equation(s,template,animate){
    cancelEffect(s.element);
    const previous=s.source.children.length?s.source.cloneNode(true):null;
    s.source.replaceChildren(template.content.cloneNode(true));sizeEquation(s);
    if(!animate||reduced.matches)return;
    if(previous){previous.classList.add('formula-ghost');previous.setAttribute('aria-hidden','true');s.element.append(previous);}
    const tween=gsap.timeline({onComplete:()=>{previous?.remove();s.source.style.opacity='1';effects.delete(s.element);}});effects.set(s.element,tween);
    if(previous)tween.to(previous,{opacity:0,duration:.22,ease:'power1.inOut'},0);
    tween.fromTo(s.source,{opacity:0},{opacity:1,duration:.32,ease:'power1.inOut'},previous?.12:0);
  }
  function fit(){
    content.style.transform='none';
    const css=getComputedStyle(page),available=Math.max(1,page.clientHeight-parseFloat(css.paddingTop)-parseFloat(css.paddingBottom));
    // The final chapter uses actual smaller type/diagram dimensions, never a
    // transform on the chapter. Resolve its responsive sizing before sliding in.
    content.style.setProperty('--forms-unit','1');
    function fitDerivations(){
      content.querySelectorAll('.formula-full-derivation').forEach(box=>{
        const math=box.querySelector('math');math.style.transform='none';math.style.removeProperty('font-size');box.style.height='auto';
        if(!box.clientWidth)return;
        const width=math.getBoundingClientRect().width;
        if(width>box.clientWidth-4)math.style.fontSize=parseFloat(getComputedStyle(math).fontSize)*(box.clientWidth-4)/width+'px';
        box.style.height=math.getBoundingClientRect().height+'px';
      });
    }
    fitDerivations();
    if(chapter>=formsChapter){
      let unit=1;
      for(let i=0;i<3&&content.offsetHeight>available;i++){
        unit*=available/content.offsetHeight;
        content.style.setProperty('--forms-unit',String(unit));fitDerivations();
      }
      viewport.style.height=content.offsetHeight+'px';
    }else{
      const scale=Math.min(1,available/Math.max(1,content.offsetHeight));content.style.transform=`scale(${scale})`;viewport.style.height=content.offsetHeight*scale+'px';
    }
    sizeEquation(story);
  }
  let chapterTransition=null,outgoing=null;
  function finishChapterTransition(){
    chapterTransition?.kill();chapterTransition=null;outgoing?.remove();outgoing=null;
    page.style.setProperty('--chapter-shift','0px');
  }
  function snapshotChapter(){
    const layer=document.createElement('div');layer.className='formula-chapter-outgoing';layer.setAttribute('aria-hidden','true');layer.inert=true;
    const bounds=page.getBoundingClientRect();
    for(const element of [heading,viewport]){
      const rect=element.getBoundingClientRect(),copy=element.cloneNode(true);
      // Freeze the outgoing chapter before its parent's chapter class changes.
      // Otherwise the final chapter's smaller fonts and heights restyle the preceding chapter mid-slide.
      const properties=['display','position','top','right','bottom','left','width','height','min-width','max-width','min-height','max-height','font-size','font-family','font-weight','font-style','line-height','letter-spacing','margin','padding','gap','row-gap','column-gap','grid-template-columns','grid-column','grid-row','align-items','align-self','justify-content','flex','order','transform','transform-origin','overflow','box-sizing'];
      const originals=[element,...element.querySelectorAll('*')],copies=[copy,...copy.querySelectorAll('*')];
      originals.forEach((original,i)=>{const css=getComputedStyle(original);for(const property of properties)copies[i].style.setProperty(property,css.getPropertyValue(property));});
      copy.style.cssText=`position:absolute;left:${rect.left-bounds.left}px;top:${rect.top-bounds.top}px;width:${rect.width}px;height:${rect.height}px;min-height:0;transform:none;margin:0;`;
      if(element===heading){const h=copy.querySelector('h2');h.style.font=getComputedStyle(title).font;h.style.margin='0';}
      for(const el of [copy,...copy.querySelectorAll('[id]')])el.removeAttribute('id');
      // Only cloned gradients need new IDs; leave the live diagrams untouched.
      const ids=new Map();
      element.querySelectorAll('linearGradient[id]').forEach((el,i)=>{const id='outgoing-gradient-'+i;ids.set(el.id,id);copy.querySelectorAll('linearGradient')[i]?.setAttribute('id',id);});
      copy.querySelectorAll('[fill]').forEach(el=>{for(const [from,to] of ids)if(el.getAttribute('fill')===`url(#${from})`)el.setAttribute('fill',`url(#${to})`);});
      layer.append(copy);
    }
    page.append(layer);return layer;
  }
  function setChapter(index,animate=true){
    const previous=chapter,direction=Math.sign(index-previous)||1;
    finishChapterTransition();
    const slide=animate&&!reduced.matches&&index!==previous;
    if(slide)outgoing=snapshotChapter();
    chapter=Math.max(0,Math.min(index,chapterCount()-1));index=chapter;const template=storyTemplates[Math.min(index,formsChapter)];
    cancelEffect(heading);title.textContent=template.dataset.title;
    cardsElement.hidden=index<formsChapter;page.classList.toggle('show-formulations',index>=formsChapter);
    page.classList.toggle('single-form',mobileForms.matches&&index>=formsChapter);
    cards.forEach((card,i)=>{card.el.hidden=mobileForms.matches&&index>=formsChapter&&i!==index-formsChapter;});page.classList.toggle('show-expectation',index===1);
    updateTemporal(index,false);
    window.alignPaperHeadings?.();equation(story,template,false);fit();
    page.querySelectorAll('[data-formula-chapter]').forEach(b=>b.setAttribute('aria-current',Number(b.dataset.formulaChapter)===index?'step':'false'));
    if(slide){
      const distance=page.clientWidth;
      chapterTransition=gsap.timeline({onComplete:finishChapterTransition});
      chapterTransition.fromTo(page,{'--chapter-shift':`${direction*distance}px`},{'--chapter-shift':'0px',duration:.65,ease:'power2.inOut'},0)
        .to(outgoing,{x:-direction*distance,duration:.65,ease:'power2.inOut'},0);
    }
  }
  function pause(){paused=true;master?.pause();effects.forEach(e=>e.pause());pauseButton.textContent='Play';}
  function start(from=0){
    finishChapterTransition();master?.kill();[...effects.keys()].forEach(cancelEffect);paused=false;pauseButton.textContent='Pause';setChapter(reduced.matches?formsChapter:from,!reduced.matches);
    if(reduced.matches||!window.gsap){pauseButton.textContent='Pause';return;}
    master=gsap.timeline({onComplete:()=>{pauseButton.textContent='Pause';}});
    const times=chapterTimes();
    times.forEach((time,i)=>{if(i>from)master.call(()=>setChapter(i),[],time-times[from]);});
    master.to({}, {duration:2},'>');
  }

  function renderChapterButtons(){
    const nav=page.querySelector('.formula-chapters');
    const names=['RoPE','Expectation and substitution','General form',...(mobileForms.matches?['Uniform','Gaussian','Learned']:['Three forms'])];
    nav.replaceChildren(...names.map((name,i)=>{
      const button=document.createElement('button');button.type='button';button.dataset.formulaChapter=String(i);
      button.setAttribute('aria-label',`Derivation step ${i+1}: ${name}`);
      const number=document.createElement('span');number.textContent=String(i+1).padStart(2,'0');button.append(number);
      button.onclick=()=>{pause();setChapter(i);};return button;
    }));
  }
  renderChapterButtons();
  mobileForms.addEventListener('change',()=>{pause();renderChapterButtons();chapter=Math.min(chapter,chapterCount()-1);entryChapter=Math.min(entryChapter,chapterCount()-1);setChapter(chapter,false);});
  page.querySelector('[data-formula-replay]').onclick=()=>start(0);
  pauseButton.onclick=()=>{if(!master||master.progress()===1){start();return;}if(paused){paused=false;master.play();effects.forEach(e=>e.play());pauseButton.textContent='Pause';}else pause();};
  window.formulationNavigation={
    pause,
    canStep:direction=>chapter+direction>=0&&chapter+direction<chapterCount(),
    step(direction){if(!this.canStep(direction))return false;pause();setChapter(chapter+direction);entryChapter=chapter;if(!active)manualEntry=true;return true;},
    prepare:direction=>{if(!active){entryChapter=direction<0?chapterCount()-1:0;manualEntry=false;setChapter(entryChapter,false);}}
  };
  setChapter(0,false);
  new IntersectionObserver(entries=>{const visible=entries[0].isIntersecting;if(visible&&!active){active=true;if(!manualEntry)start(entryChapter);manualEntry=false;}else if(!visible&&active){active=false;pause();}},{threshold:.5}).observe(page);
  new ResizeObserver(()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{finishChapterTransition();[...effects.keys()].forEach(cancelEffect);gsap.set(timeDiagram.distribution,{scaleX:chapter===0?0:1,opacity:chapter===0?0:1});gsap.set(timeDiagram.marker,{opacity:chapter===0?1:0});gsap.set(cards.map(c=>c.distribution),{scaleY:1,opacity:1});fit();},120);}).observe(page);
  document.fonts.ready.then(fit);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
  reduced.addEventListener('change',()=>{if(reduced.matches){master?.kill();setChapter(formsChapter,false);}});
})();
