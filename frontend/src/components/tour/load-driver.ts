/**
 * driver.js and its stylesheet, fetched only when a tour starts, so neither
 * sits on the console's critical path.
 */
import "driver.js/dist/driver.css";

export { driver } from "driver.js";
export type { Driver, DriveStep, PopoverDOM } from "driver.js";
