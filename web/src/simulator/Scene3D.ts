// three.js model of the planned CoreXY pen plotter, in millimetres.
// World axes: x to the right, y up, z towards the viewer. The machine origin
// (X0 Y0) is the front-left corner of the A4 sheet; machine +Y points to the back.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export type ViewMode = "build" | "exploded" | "plot" | "top";

export type BuildStep = { id: string; title: string; text: string; parts: string };

export const BUILD_STEPS: BuildStep[] = [
  { id: "base", title: "Base board", text: "A 460 × 400 mm board is the frame's foundation and the writing surface.", parts: "12 mm plywood" },
  { id: "railL", title: "Left Y rail", text: "A 2020 aluminium extrusion on two corner posts carries the left end of the gantry.", parts: "2020 extrusion 400 mm, 2 posts" },
  { id: "railR", title: "Right Y rail", text: "The matching right rail; both must be parallel so the gantry does not bind.", parts: "2020 extrusion 400 mm, 2 posts" },
  { id: "railBack", title: "Back cross rail", text: "Ties the Y rails together and holds both motors.", parts: "2020 extrusion 400 mm, corner brackets" },
  { id: "motorA", title: "Motor A", text: "NEMA 17 stepper at the back left with a 20-tooth GT2 pulley.", parts: "NEMA 17, GT2 20T pulley" },
  { id: "motorB", title: "Motor B", text: "NEMA 17 stepper at the back right. In CoreXY both motors move for every X or Y move.", parts: "NEMA 17, GT2 20T pulley" },
  { id: "idlers", title: "Idler pulleys", text: "Back idlers turn the crossed belts around the corners.", parts: "2 × GT2 idlers" },
  { id: "sliders", title: "Gantry sliders", text: "Printed blocks with V-wheels ride on the Y rails and carry the gantry idlers.", parts: "2 printed blocks, 8 V-wheels" },
  { id: "gantry", title: "Gantry (X rail)", text: "A 2020 extrusion across the sliders. The carriage runs along it.", parts: "2020 extrusion 420 mm" },
  { id: "carriage", title: "Pen carriage", text: "A printed carriage with V-wheels. Both belt ends clamp to it.", parts: "Printed carriage, 4 V-wheels" },
  { id: "servo", title: "Pen-lift servo", text: "An SG90 servo lifts and lowers the pen (M5 up, M3 down).", parts: "SG90 micro servo" },
  { id: "pen", title: "Pen and holder", text: "A spring-loaded holder keeps even pressure on the paper.", parts: "Gel pen, printed holder, spring" },
  { id: "electronics", title: "Controller", text: "Arduino Uno with CNC shield and two A4988 drivers runs GRBL and reads the G-code.", parts: "Arduino Uno, CNC shield, 2 × A4988, 12 V supply" },
  { id: "paper", title: "Paper", text: "An A4 sheet, clipped at the corners. The machine origin is its front-left corner.", parts: "A4 sheet, 4 clips" },
];

const PAPER_W = 210;
const PAPER_H = 297;
const PAPER_X0 = -105; // world x of machine X = 0
const PAPER_Z0 = 148.5; // world z of machine Y = 0
const RAIL_X = 200;
const RAIL_Y = 50; // centre height of the Y rails
const BACK_Z = -185;
const GANTRY_Y = 75;
const PEN_OFFSET_Z = 24; // pen tip is this far in front of the gantry rail
const PAPER_TOP = 0.4;

const EXPLODE: Record<string, [number, number, number]> = {
  base: [0, -1.2, 0], railL: [-1, 0.2, 0], railR: [1, 0.2, 0], railBack: [0, 0.3, -1],
  motorA: [-0.8, 0.6, -1], motorB: [0.8, 0.6, -1], idlers: [0, 0.8, -1.2], sliders: [0, 1.0, 0.3],
  gantry: [0, 1.4, 0], carriage: [0, 1.9, 0.6], servo: [0.7, 2.2, 0.8], pen: [0, 2.6, 1.2],
  electronics: [1.2, 0.3, 1], paper: [0, -0.5, 1.4],
};

const mat = {
  alu: new THREE.MeshStandardMaterial({ color: 0xb9c0c8, metalness: 0.75, roughness: 0.35 }),
  wood: new THREE.MeshStandardMaterial({ color: 0xc8a57a, roughness: 0.85 }),
  motor: new THREE.MeshStandardMaterial({ color: 0x24272c, metalness: 0.4, roughness: 0.5 }),
  motorCap: new THREE.MeshStandardMaterial({ color: 0x8b9097, metalness: 0.8, roughness: 0.3 }),
  printed: new THREE.MeshStandardMaterial({ color: 0xe8743b, roughness: 0.6 }),
  servo: new THREE.MeshStandardMaterial({ color: 0x2f5fd0, roughness: 0.5 }),
  pcb: new THREE.MeshStandardMaterial({ color: 0x0f7a5c, roughness: 0.6 }),
  black: new THREE.MeshStandardMaterial({ color: 0x15171a, roughness: 0.7 }),
  pen: new THREE.MeshStandardMaterial({ color: 0x1d2a6b, roughness: 0.4 }),
  white: new THREE.MeshStandardMaterial({ color: 0xf2f2f2, roughness: 0.5 }),
};

function box(w: number, h: number, d: number, m: THREE.Material, x = 0, y = 0, z = 0) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m);
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function cyl(r: number, h: number, m: THREE.Material, x = 0, y = 0, z = 0) {
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, 24), m);
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  return mesh;
}

function motor(x: number, z: number) {
  const g = new THREE.Group();
  g.add(box(42, 40, 42, mat.motor, 0, 0, 0));
  g.add(box(42.5, 6, 42.5, mat.motorCap, 0, 17, 0));
  g.add(box(42.5, 6, 42.5, mat.motorCap, 0, -17, 0));
  g.add(cyl(6.5, 14, mat.motorCap, 0, 27, 0));
  g.position.set(x, RAIL_Y + 30, z);
  return g;
}

export class PlotterScene {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private groups: Record<string, THREE.Group> = {};
  private home: Record<string, THREE.Vector3> = {};
  private beltA: THREE.Line;
  private beltB: THREE.Line;
  private servoArm: THREE.Mesh;
  private paperTex: THREE.CanvasTexture | null = null;
  private paperMesh: THREE.Mesh;
  private mode: ViewMode = "plot";
  private buildStep = BUILD_STEPS.length;
  private dropStart = 0;
  private explode = 0;
  private machine = { x: 0, y: 0, pen: false };
  private penLift = 1; // 0 = down, 1 = up; eased towards target
  private raf = 0;
  private resizeObs: ResizeObserver;

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(this.renderer.domElement);

    this.scene.background = new THREE.Color(0x0e1116);
    this.camera = new THREE.PerspectiveCamera(40, 1, 1, 5000);
    this.camera.position.set(420, 520, 620);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 30, 0);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.49;

    this.scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x2a2018, 0.9));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(300, 700, 400);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -400, right: 400, top: 400, bottom: -400, near: 100, far: 1600 });
    this.scene.add(sun);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(4000, 4000), new THREE.ShadowMaterial({ opacity: 0.35 }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -12.5;
    floor.receiveShadow = true;
    this.scene.add(floor);
    const grid = new THREE.GridHelper(1600, 32, 0x2a3240, 0x1a202a);
    grid.position.y = -12.4;
    this.scene.add(grid);

    // ---- parts ----
    const add = (id: string, g: THREE.Group) => {
      this.groups[id] = g;
      this.home[id] = g.position.clone();
      this.scene.add(g);
    };
    const group = (...objs: THREE.Object3D[]) => {
      const g = new THREE.Group();
      objs.forEach((o) => g.add(o));
      return g;
    };

    add("base", group(box(460, 12, 400, mat.wood, 0, -6, 0)));
    const rail = (x: number) =>
      group(box(20, 20, 400, mat.alu, x, RAIL_Y, 0), box(20, RAIL_Y - 10, 20, mat.alu, x, (RAIL_Y - 10) / 2, -185),
        box(20, RAIL_Y - 10, 20, mat.alu, x, (RAIL_Y - 10) / 2, 185));
    add("railL", rail(-RAIL_X));
    add("railR", rail(RAIL_X));
    add("railBack", group(box(380, 20, 20, mat.alu, 0, RAIL_Y, BACK_Z)));
    add("motorA", group(motor(-RAIL_X, BACK_Z)));
    add("motorB", group(motor(RAIL_X, BACK_Z)));
    add("idlers", group(cyl(8, 10, mat.motorCap, -RAIL_X + 22, RAIL_Y + 22, BACK_Z), cyl(8, 10, mat.motorCap, RAIL_X - 22, RAIL_Y + 22, BACK_Z)));

    const sliders = new THREE.Group();
    for (const x of [-RAIL_X, RAIL_X]) {
      sliders.add(box(34, 24, 60, mat.printed, x, RAIL_Y + 18, 0));
      sliders.add(cyl(8, 10, mat.motorCap, x, GANTRY_Y + 14, 0));
    }
    add("sliders", sliders);
    add("gantry", group(box(420, 20, 20, mat.alu, 0, GANTRY_Y, 0)));

    const carriage = new THREE.Group();
    carriage.add(box(56, 34, 36, mat.printed, 0, GANTRY_Y, 6));
    carriage.add(box(22, 60, 14, mat.printed, 0, GANTRY_Y - 10, PEN_OFFSET_Z - 2));
    add("carriage", carriage);

    const servo = new THREE.Group();
    servo.add(box(23, 22, 12, mat.servo, 24, GANTRY_Y + 18, PEN_OFFSET_Z));
    this.servoArm = box(4, 3, 18, mat.white, 18, GANTRY_Y + 30, PEN_OFFSET_Z + 6);
    servo.add(this.servoArm);
    add("servo", servo);

    const pen = new THREE.Group();
    const barrel = cyl(4.5, 100, mat.pen, 0, 50 + 4, 0);
    const tip = new THREE.Mesh(new THREE.ConeGeometry(4.5, 8, 24), mat.black);
    tip.rotation.x = Math.PI;
    tip.position.y = 4;
    pen.add(barrel, tip);
    pen.position.set(0, PAPER_TOP, PEN_OFFSET_Z);
    add("pen", pen);

    // Controller and supply sit behind the sheet, under the back rail's reach.
    add("electronics", group(box(70, 4, 40, mat.pcb, 60, 2, -172), box(70, 14, 40, mat.black, 60, 11, -172),
      box(50, 22, 40, mat.motorCap, -60, 11, -172)));

    this.paperMesh = new THREE.Mesh(new THREE.PlaneGeometry(PAPER_W, PAPER_H), new THREE.MeshStandardMaterial({ color: 0xfbfaf6, roughness: 0.9 }));
    this.paperMesh.rotation.x = -Math.PI / 2;
    this.paperMesh.position.set(PAPER_X0 + PAPER_W / 2, PAPER_TOP, PAPER_Z0 - PAPER_H / 2);
    this.paperMesh.receiveShadow = true;
    const clips = [0, 1].flatMap((i) => [0, 1].map((j) =>
      box(14, 4, 14, mat.black, PAPER_X0 + i * PAPER_W, PAPER_TOP + 2, PAPER_Z0 - j * PAPER_H)));
    add("paper", group(this.paperMesh, ...clips));

    const beltMat = (c: number) => new THREE.LineBasicMaterial({ color: c });
    this.beltA = new THREE.Line(new THREE.BufferGeometry(), beltMat(0xff6b6b));
    this.beltB = new THREE.Line(new THREE.BufferGeometry(), beltMat(0x5ad1ff));
    this.scene.add(this.beltA, this.beltB);

    this.resizeObs = new ResizeObserver(() => this.resize());
    this.resizeObs.observe(host);
    this.resize();
    this.applyMachine();
    const loop = () => {
      this.raf = requestAnimationFrame(loop);
      this.frame();
    };
    loop();
  }

  setPaperCanvas(canvas: HTMLCanvasElement) {
    this.paperTex?.dispose();
    this.paperTex = new THREE.CanvasTexture(canvas);
    this.paperTex.colorSpace = THREE.SRGBColorSpace;
    this.paperTex.anisotropy = 8;
    const m = this.paperMesh.material as THREE.MeshStandardMaterial;
    m.map = this.paperTex;
    m.color.set(0xffffff);
    m.needsUpdate = true;
  }

  markPaperDirty() {
    if (this.paperTex) this.paperTex.needsUpdate = true;
  }

  setMode(mode: ViewMode) {
    const prev = this.mode;
    this.mode = mode;
    if (mode === "top") {
      this.camera.position.set(0, 720, 1);
      this.controls.target.set(0, 0, 0);
    } else if (prev === "top") {
      this.camera.position.set(420, 520, 620);
      this.controls.target.set(0, 30, 0);
    }
    if (mode !== "build") this.buildStep = BUILD_STEPS.length;
    if (mode !== "exploded") this.explode = 0;
  }

  setBuildStep(n: number) {
    if (n !== this.buildStep) this.dropStart = performance.now();
    this.buildStep = Math.max(0, Math.min(BUILD_STEPS.length, n));
  }

  setExplode(f: number) {
    this.explode = f;
  }

  /** Machine coordinates in mm (origin front-left of the sheet, Y towards the back). */
  setMachine(x: number, y: number, pen: boolean) {
    this.machine = { x, y, pen };
  }

  snapshot(): string {
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL("image/png");
  }

  dispose() {
    cancelAnimationFrame(this.raf);
    this.resizeObs.disconnect();
    this.controls.dispose();
    this.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      m.geometry?.dispose?.();
    });
    this.paperTex?.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  private resize() {
    const w = this.host.clientWidth || 1;
    const h = this.host.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private applyMachine() {
    const target = this.machine.pen ? 0 : 1;
    this.penLift += (target - this.penLift) * 0.35;
    // In top view the gantry is parked behind the sheet so the whole page shows.
    const parked = this.mode === "top";
    const wx = PAPER_X0 + (parked ? PAPER_W / 2 : this.machine.x);
    const wz = PAPER_Z0 - (parked ? PAPER_H + 8 : this.machine.y) - PEN_OFFSET_Z;
    const g = this.groups;
    const ex = (id: string) => {
      const d = EXPLODE[id];
      return new THREE.Vector3(d[0], d[1], d[2]).multiplyScalar(this.explode * 140);
    };
    for (const id of ["sliders", "gantry"]) g[id].position.copy(this.home[id]).add(new THREE.Vector3(0, 0, wz)).add(ex(id));
    for (const id of ["carriage", "servo"]) g[id].position.copy(this.home[id]).add(new THREE.Vector3(wx, 0, wz)).add(ex(id));
    g.pen.position.copy(this.home.pen).add(new THREE.Vector3(wx, this.penLift * 7, wz)).add(ex("pen"));
    this.servoArm.rotation.y = this.penLift * 0.9;
    for (const id of Object.keys(g)) {
      if (["sliders", "gantry", "carriage", "servo", "pen"].includes(id)) continue;
      g[id].position.copy(this.home[id]).add(ex(id));
    }

    // Build mode: show parts up to the current step; the newest one drops in.
    const n = this.mode === "build" ? this.buildStep : BUILD_STEPS.length;
    const k = Math.min(1, (performance.now() - this.dropStart) / 600);
    const ease = 1 - Math.pow(1 - k, 3);
    BUILD_STEPS.forEach((s, i) => {
      g[s.id].visible = i < n;
      if (this.mode === "build" && i === n - 1) g[s.id].position.y += (1 - ease) * 160;
    });

    // Crossed CoreXY belts, drawn through the pulley centres.
    const showBelts = n >= BUILD_STEPS.findIndex((s) => s.id === "carriage") + 1 && this.explode < 0.05;
    this.beltA.visible = this.beltB.visible = showBelts;
    if (showBelts) {
      const gz = wz;
      const yA = GANTRY_Y + 12, yB = GANTRY_Y + 16;
      const P = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
      this.beltA.geometry.setFromPoints([
        P(wx - 20, yA, gz), P(-RAIL_X, yA, gz), P(-RAIL_X, yA, BACK_Z), P(RAIL_X - 22, yA, BACK_Z),
        P(RAIL_X, yA, BACK_Z + 10), P(RAIL_X, yA, gz), P(wx + 20, yA, gz),
      ]);
      this.beltB.geometry.setFromPoints([
        P(wx + 20, yB, gz), P(RAIL_X, yB, gz), P(RAIL_X, yB, BACK_Z), P(-RAIL_X + 22, yB, BACK_Z),
        P(-RAIL_X, yB, BACK_Z + 10), P(-RAIL_X, yB, gz), P(wx - 20, yB, gz),
      ]);
    }
  }

  private frame() {
    this.applyMachine();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
