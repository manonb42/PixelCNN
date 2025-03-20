export type Vec2D = { x: number, y: number }
export type Color = number[]

function vec2d_eq(a: Vec2D, b: Vec2D): boolean {
  return a.x === b.x && a.y === b.y;
}

export class BitCanvas {

  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private grid_size: Vec2D = {x:0, y:0};
  private pixel_size: Vec2D = {x:0, y:0};
  private currentColor: Color = [255, 255, 255]

  private marker_color: Color = [255, 0, 0]
  private marker: Vec2D = {x:0, y:0}

  private buffer: Color[][] = [];

  private mouse_down: boolean = false;
  private mouse_action: 'marker' | 'draw' = 'draw'

  constructor(canvas: HTMLCanvasElement, grid_width: number, grid_height: number) {
    this.canvas = canvas;

    const context = this.canvas.getContext('2d');
    if (!context) {
      throw new Error('Unable to obtain 2D context from canvas.');
    }
    this.ctx = context;
    this.ctx.imageSmoothingEnabled = false;


    // Attach mouse event listeners.
    this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
    this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
    window.addEventListener('mouseup', this.onMouseUp.bind(this));
    this.canvas.addEventListener('mouseleave', this.onMouseUp.bind(this));

    this.emptyBuffer()

    this.updateSize(grid_width, grid_height) // also calls reset
  }

  updateSize(grid_width: number, grid_height: number){
    this.grid_size = { x: grid_width, y: grid_height };
    this.pixel_size = {
        x: Math.floor(this.canvas.width / grid_width),
        y: Math.floor(this.canvas.height / grid_height)
    };
    this.reset()
  }

  inbounds(pos: Vec2D): boolean {
    return pos.x >= 0 && pos.x < this.grid_size.x && pos.y >= 0 && pos.y < this.grid_size.y;
  }

  // Sets a pixel in the buffer and draws it on the canvas.
  setPixel(pos: Vec2D, color: Color): void {
    if (!this.inbounds(pos)) return;
    this.buffer[pos.y][pos.x] = color;
    this.drawPixel(pos, color);
  }



  private emptyBuffer(): void {
    // Initialize the buffer with white pixels.
    this.buffer = [];
    for (let y = 0; y < this.grid_size.y; y++) {
      this.buffer[y] = [];
      for (let x = 0; x < this.grid_size.x; x++) {
        this.buffer[y][x] = [0, 0, 0];
      }
    }
  }

  // Loads a buffer with shape [grid_height, grid_width, 3] and redraws the canvas.

  loadBuffer(buf: Color[][]): void {
    if (buf.length !== this.grid_size.y || buf[0].length !== this.grid_size.x) {
      throw new Error('Buffer dimensions do not match the canvas grid dimensions.');
    }
    this.buffer = buf;
    this.redraw();
  }

  // Returns a deep copy of the current buffer.
  saveBuffer(): Color[][] {
    return this.buffer.map(row => row.map(pixel => [...pixel]));
  }
  getMarker(): Vec2D {
    return this.marker
  }

  // Sets the current drawing color.
  setColor(color: Color): void {
    this.currentColor = color;
  }

  // Draws a single pixel on the canvas at the given grid coordinates.
  drawPixel(pos: Vec2D, color: Color): void {
    if (vec2d_eq(pos, this.marker)) { color = this.marker_color}
    this.ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    this.ctx.fillRect(
      pos.x * this.pixel_size.x,
      pos.y * this.pixel_size.y,
      this.pixel_size.x,
      this.pixel_size.y
    );
  }

  // Redraws the entire canvas from the buffer.
  reset(): void {
    this.emptyBuffer();
    this.redraw();
  }

  redraw(): void {
    for (let y = 0; y < this.grid_size.y; y++) {
      for (let x = 0; x < this.grid_size.x; x++) {
        this.drawPixel({x, y}, this.buffer[y][x]);
      }
    }
  }

  private getMousePos(ev: MouseEvent): Vec2D {
    const rect = this.canvas.getBoundingClientRect();
    const scale_x = this.canvas.width / rect.width;
    const scale_y = this.canvas.height / rect.height;
    const canvas_x = (ev.clientX - rect.left) * scale_x;
    const canvas_y = (ev.clientY - rect.top) * scale_y;
    return {
      x: Math.floor(canvas_x / this.pixel_size.x),
      y: Math.floor(canvas_y / this.pixel_size.y)
    };
  }

  private doMouseAction(pos: Vec2D): void {
      if (!this.inbounds(pos)) return;
      if (this.mouse_action === 'marker') {
        if (vec2d_eq(pos, this.marker)) return;
        const old = {...this.marker};
        this.marker = pos;
        this.drawPixel(this.marker, this.marker_color);
        this.drawPixel(old, this.buffer[old.y][old.x]);
      } else {
        this.setPixel(pos, this.currentColor);
      }
  }
  private onMouseDown(ev: MouseEvent): void {
    if (this.mouse_action === 'marker') {
      this.mouse_action = 'draw';
      return;
    }
    this.mouse_down = true;
    const pos = this.getMousePos(ev);
    if (vec2d_eq(pos, this.marker)) { this.mouse_action = 'marker' }
    else { this.mouse_action = 'draw' }
    this.doMouseAction(pos);
  }

  private onMouseMove(ev: MouseEvent): void {
    if (!this.mouse_down && this.mouse_action === 'draw') return;
    const pos = this.getMousePos(ev);
    this.doMouseAction(pos);
  }

  private onMouseUp(_ev: MouseEvent): void {
    this.mouse_down = false;
  }
}


export function antialias_buffer(buffer: Color[][]): Color[][] {
  const out = [];

  const h = buffer.length;
  const w = buffer[0].length;

  for (let y = 0; y < h; y++) {
        out[y] = [] as Color[];
        for (let x = 0; x < w; x++) {
          let sum_r = 0, sum_g = 0, sum_b = 0, count = 0;
          for (let j = -1; j <= 1; j++) {
            for (let i = -1; i <= 1; i++) {
              const nx = x + i;
              const ny = y + j;
              if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                const pixel = buffer[ny][nx];
                sum_r += pixel[0];
                sum_g += pixel[1];
                sum_b += pixel[2];
                count++;
              }
            }
          }
          out[y][x] = [
            Math.max(buffer[y][x][0], Math.round(sum_r / count / 3)),
            Math.max(buffer[y][x][1], Math.round(sum_g / count / 3)),
            Math.max(buffer[y][x][2], Math.round(sum_b / count / 3)),
          ];
        }
      }
  return out;
}
