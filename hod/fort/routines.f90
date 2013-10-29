!! FORTRAN ROUTINES TO DO QUICK WORK ON LOOPS FOR HOD
module hod_routines
contains
subroutine simps(dx, func,val)
    implicit none
    !!simps works on functions defined at EQUIDISTANT values of x only

    real(8), intent(in) :: dx !The grid spacing
    real(8), intent(in) :: func(:) !The function
    real(8), intent(out) :: val !The value of the integral

    integer :: n !length of func
    integer :: i

    n = size(func)
    if (mod(n,2)>0) stop "n must be even"

    val = func(1) + func(n) + 4*func(n-1)
    do i=1,n/2-1
        val = val + 2*func(2*i)
        val = val + 4*func(2*i-1)
    end do
    val = val*dx/3
end subroutine

subroutine trapz(dx,func,val)
    implicit none
    !!trapz works on functions defined at EQUIDISTANT values of x only

    real(8), intent(in) :: dx !The grid spacing
    real(8), intent(in) :: func(:) !The function
    real(8), intent(out) :: val !The value of the integral

    integer :: n !length of func
    integer :: i

    val = func(1)+func(n)
    do i=2,n-1
        val = val + 2*func(i)
    end do
    val = val*dx/2
end subroutine

!!!!!!!!!!!!! START ACTUAL CALLED ROUTINES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine power_gal_2h(nlnk,nm,u,bias,ntot,dndm,mass,power)
    implicit none

    integer, intent(in) :: nlnk, nm !Size of lnk and mass arrays respectively
    real(8), intent(in) :: bias(nm), ntot(nm),dndm(nm), mass(nm)
    real(8), intent(in) :: u(nlnk,nm)

    real(8), intent(out):: power(nlnk)

    integer :: i
    real(8) :: integrand(nm),dm
    integrand = 0.d0
    dm = log(mass(1))-log(mass(2))

    do i=1,nlnk
        integrand = ntot * u(i,:) *dndm*bias*mass !multiply by mass as we integrate in log-space
        call simps(dm,integrand,power(i))
    end do

end subroutine

subroutine power_gal_1h_ss(nlnk,nm,u,dndm,nsat,ncen,mass,central,power)
    implicit none

    integer, intent(in) :: nlnk, nm !Size of lnk and mass arrays respectively
    real(8), intent(in) :: ncen(nm),nsat(nm),dndm(nm), mass(nm)
    real(8), intent(in) :: u(nlnk,nm)
    logical, intent(in) :: central

    real(8), intent(out):: power(nlnk)

    integer :: i
    real(8) :: integrand(nm),dm
    integrand = 0.d0

    dm = log(mass(2))-log(mass(1))

    do i=1,nlnk
        if (central)then
            integrand = ncen*nsat**2 * u(i,:)**2*dndm*mass
        else
            integrand = nsat**2 * u(i,:)**2 *dndm*mass
        end if
        call simps(dm,integrand,power(i))
    end do
end subroutine

subroutine corr_gal_1h_ss(nr,nm,mass,dndm,ncen,nsat,lam,central,corr)
    implicit none

    integer, intent(in) :: nr,nm
    real(8), intent(in) :: mass(nm), ncen(nm),nsat(nm),dndm(nm)
    real(8), intent(in) :: lam(nr,nm)
    logical, intent(in) :: central

    real(8), intent(out) :: corr(nr)

    integer :: i
    real(8) :: integrand(nm),dm
    integrand = 0.d0
    dm = log(mass(2))-log(mass(1))

    do i=1,nr
        if (central)then
            integrand = dndm*ncen*nsat**2*lam(i,:)/mass
        else
            integrand = dndm*nsat**2*lam(i,:)/mass
        end if

        call simps(dm,integrand,corr(i))
    end do
end subroutine

subroutine corr_gal_1h_cs(nr,nm,r,mass,dndm,ncen,nsat,rho,mean_dens,delta_halo,corr)
    implicit none

    integer, intent(in) :: nr,nm
    real(8), intent(in) :: r(nr), mass(nm), ncen(nm),nsat(nm),dndm(nm)
    real(8), intent(in) :: rho(nr,nm),mean_dens,delta_halo

    real(8), intent(out) :: corr(nr)

    integer :: i
    real(8) :: mmin
    real(8) :: integrand(nm),dm
    integrand = 0.d0

    dm = log(mass(2))-log(mass(1))

    do i=1,nr
        mmin = 4*3.14159265*r(i)**3*mean_dens*delta_halo/3
        where (mass>mmin) integrand = dndm*2*ncen*nsat*rho(i,:)
        call simps(dm,integrand,corr(i))
    end do
end subroutine

subroutine corr_gal_1h(nr,nm,r,mass,dndm,ncen,nsat,rho,lam,central,mean_dens,delta_halo,corr)
    implicit none

    integer, intent(in) :: nr,nm
    real(8), intent(in) :: r(nr), mass(nm), ncen(nm),nsat(nm),dndm(nm)
    real(8), intent(in) :: rho(nr,nm),mean_dens,delta_halo
    real(8), intent(in) :: lam(nr,nm)
    logical, intent(in) :: central

    real(8), intent(out) :: corr(nr)

    integer :: i
    real(8) :: mmin
    real(8) :: integrand(nm),dm


    integrand = 0.d0
    dm = log(mass(2))-log(mass(1))

    do i=1,nr
        if (central)then
            integrand = dndm*ncen*nsat**2*lam(i,:)/mass
        else
            integrand = dndm*nsat**2*lam(i,:)/mass
        end if

        mmin = 4*3.14159265*r(i)**3*mean_dens*delta_halo/3
        where (mass>mmin) integrand = integrand+dndm*2*ncen*nsat*rho(i,:)
        call simps(dm,integrand,corr(i))
    end do
end subroutine

subroutine power_to_corr(nlnk,nr,lnk,r,power,corr)
    implicit none
    !This routine is a bit of a hack really. We stop integrating when
    !we can no longer fit 5 dlnk within an oscillation.
    !There is no saying that the value we end up on should be the correct one.
    !If maxk is large, then we should be fine, but if not its a bit luck-of-the-draw.
    integer, intent(in) :: nlnk, nr
    real(8), intent(in) :: r(nr),power(nlnk),lnk(nlnk)

    real(8), intent(out) :: corr(nr)

    integer :: i
    real(8) :: k(nlnk), integrand(nlnk),dlnk,maxk
    integrand = 0.d0

    k = exp(lnk)
    dlnk = lnk(2) - lnk(1)

    do i=1,nr
        maxk = 3.14159265/(5*r(i)*(exp(lnk(2)-lnk(1))-1))
        where (k < maxk) integrand = power*k**2*sin(k*r(i))/r(i)
        call simps(dlnk,integrand,corr(i))
    end do
end subroutine
end module
