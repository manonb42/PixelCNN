import { Streamlit, RenderData } from "streamlit-component-lib"
import { BitCanvas, antialias_buffer } from "./canvas"




const canvas_el = document.getElementById("canvas")
const canvas = new BitCanvas(canvas_el as any, 28, 28)

const reset_button = document.body.appendChild(document.createElement("button"))
reset_button.textContent = "Reset"
reset_button.onclick = function(): void {
    canvas.reset()
}

const antialias_button = document.body.appendChild(document.createElement("button"))
antialias_button.textContent = "Antialias"
antialias_button.onclick = function(): void {
    canvas.loadBuffer(antialias_buffer(canvas.saveBuffer()))
}


const send_button = document.body.appendChild(document.createElement("button"))
send_button.textContent = "Send!"


send_button.onclick = function(): void {
    Streamlit.setComponentValue({
        "image": canvas.saveBuffer(),
        "marker": canvas.getMarker()
    })
}


/**
 * The component's render function. This will be called immediately after
 * the component is initially loaded, and then again every time the
 * component gets new data from Python.
 */
function onRender(event: Event): void {
  // Get the RenderData from the event
  const data = (event as CustomEvent<RenderData>).detail

  canvas.updateSize(data.args["width"], data.args["height"])

  if (data.args["image"]){
      let buf = data.args["image"];
      canvas.loadBuffer(buf)
  }


  // We tell Streamlit to update our frameHeight after each render event, in
  // case it has changed. (This isn't strictly necessary for the example
  // because our height stays fixed, but this is a low-cost function, so
  // there's no harm in doing it redundantly.)
  Streamlit.setFrameHeight()
}

// Attach our `onRender` handler to Streamlit's render event.
Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)

// Tell Streamlit we're ready to start receiving data. We won't get our
// first RENDER_EVENT until we call this function.
Streamlit.setComponentReady()

// Finally, tell Streamlit to update our initial height. We omit the
// `height` parameter here to have it default to our scrollHeight.
Streamlit.setFrameHeight()
