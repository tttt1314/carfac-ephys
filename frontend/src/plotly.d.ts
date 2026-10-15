declare module 'plotly.js-dist-min' {
  const Plotly: {
    react(root: HTMLElement, data: object[], layout: object, config: object): Promise<HTMLElement>;
    purge(root: HTMLElement): void;
    Plots: { resize(root: HTMLElement): void };
  };
  export default Plotly;
}
